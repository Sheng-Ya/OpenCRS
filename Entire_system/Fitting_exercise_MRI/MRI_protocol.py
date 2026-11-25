"""Workload stages and streamed approximations to the MRI acquisition window."""

import numpy as np


def build_workload_stages(max_watts, initial_watts=0.0, final_stage_seconds=30.0):
    """Return (watts, seconds, pedalling) stages for a 25 W/min ramp."""
    values = np.asarray([max_watts, initial_watts, final_stage_seconds], dtype=float)
    if (not np.isfinite(values).all() or initial_watts < 0
            or max_watts < initial_watts or not 0 < final_stage_seconds <= 60):
        raise ValueError("Use finite nonnegative workloads and a final stage in (0, 60] seconds.")
    steps = (max_watts - initial_watts) / 25.0
    count = int(round(steps))
    if not np.isclose(steps, count, rtol=0.0, atol=1e-8):
        raise ValueError("MAX WATTS must differ from the initial workload by a multiple of 25 W.")
    # The initial minute is retained even when no higher stage was reached.
    return [(float(initial_watts + 25*index),
             float(final_stage_seconds if index == count and count else 60.0), True)
            for index in range(count + 1)]


class WindowObserver:
    """Collect complete-beat MRI targets without retaining a whole acquisition trace.

    Mean beatwise maximum/minimum total ventricular volumes approximate the scan
    operator; sequential slice acquisition and reconstruction are not represented.
    Systolic/diastolic pressures are the mean beatwise maximum/minimum of P_sa.
    Beats beginning before the window or ending after it are excluded. HR is the
    mean latched HR within each complete beat, averaged equally across those beats.
    ``peak_hr`` instead includes all HR samples inside the window, including partial
    beats. Optional gas means integrate the entire physical-time window.
    """

    _REQUIRED_KEYS = ("all_time", "time_since_beat_store", "HR_store",
                      "V_lv_store", "V_rv_store", "P_sa_store")
    _GAS_KEYS = ("Pa_O2_every_store", "Pa_CO2_every_store")

    def __init__(self, window_start, window_end):
        endpoints = np.asarray([window_start, window_end], dtype=float)
        if not np.isfinite(endpoints).all() or window_start < 0 or window_end <= window_start:
            raise ValueError("The observation window must have finite increasing nonnegative times.")
        self.window_start = float(window_start)
        self.window_end = float(window_end)
        self.peak_hr = np.nan
        self._covered_until = self.window_start
        self._last_time = -np.inf
        self._beat_id = None
        self._beat_complete_start = False
        self._beat_lv_max = self._beat_rv_max = self._beat_sa_max = -np.inf
        self._beat_lv_min = self._beat_rv_min = self._beat_sa_min = np.inf
        self._beat_hr_sum = 0.0
        self._beat_count = 0
        self._completed = []
        self._gas_integral = np.zeros(2)
        self._gas_seconds = 0.0

    def _tolerance(self):
        return max(1e-8, 1e-12*max(abs(self.window_start), abs(self.window_end)))

    def _ordered_trace(self, storage, segment_end):
        arrays = {key: np.asarray(storage[key], dtype=float) for key in self._REQUIRED_KEYS}
        capacity = len(arrays["all_time"])
        if not capacity or any(array.shape != (capacity,) for array in arrays.values()):
            raise ValueError("Observation storage arrays must have equal nonzero one-dimensional sizes.")
        writes = int(np.asarray(storage["i"]).item())
        if writes < 0:
            raise ValueError("The observation storage write count cannot be negative.")
        count = min(writes, capacity)
        indices = np.arange(writes-count, writes, dtype=np.int64) % capacity
        arrays = {key: array[indices] for key, array in arrays.items()}
        has_gases = all(key in storage for key in self._GAS_KEYS)
        if has_gases:
            for key in self._GAS_KEYS:
                gas = np.asarray(storage[key], dtype=float)
                if gas.shape != (capacity,):
                    raise ValueError("Gas storage arrays must match the observation buffer size.")
                arrays[key] = gas[indices]
        times = arrays["all_time"]
        # Unused legacy entries and rewound evaluations do not represent observations.
        valid = np.isfinite(times) & (times >= 0) & (times != 100000.0)
        valid &= times <= float(segment_end) + self._tolerance()
        if writes > capacity:
            valid &= times > 0
        arrays = {key: array[valid] for key, array in arrays.items()}
        times = arrays["all_time"]
        # Stable sorting preserves buffer order, so equal times keep their final evaluation.
        order = np.argsort(times, kind="stable")
        arrays = {key: array[order] for key, array in arrays.items()}
        times = arrays["all_time"]
        final = np.r_[times[1:] != times[:-1], True] if times.size else np.empty(0, dtype=bool)
        return {key: array[final] for key, array in arrays.items()}, has_gases

    def _start_beat(self, beat_id, complete_start):
        self._beat_id = float(beat_id)
        self._beat_complete_start = bool(complete_start)
        self._beat_lv_max = self._beat_rv_max = self._beat_sa_max = -np.inf
        self._beat_lv_min = self._beat_rv_min = self._beat_sa_min = np.inf
        self._beat_hr_sum = 0.0
        self._beat_count = 0

    def update(self, storage, segment_start, segment_end):
        """Consume the observed part of one segment; preserve pending beats across calls."""
        endpoints = np.asarray([segment_start, segment_end], dtype=float)
        if not np.isfinite(endpoints).all() or segment_end <= segment_start:
            raise ValueError("Simulation segments must have finite increasing times.")
        low = max(float(segment_start), self.window_start, self._covered_until)
        high = min(float(segment_end), self.window_end)
        tolerance = self._tolerance()
        if high <= low:
            return
        if low > self._covered_until + tolerance:
            raise ValueError("The observation window contains an unobserved time gap.")
        trace, has_gases = self._ordered_trace(storage, segment_end)
        times = trace["all_time"]
        if (times.size == 0 or times[0] > low + tolerance
                or times[-1] < high - tolerance):
            raise ValueError("The observation buffer does not cover this physical-time segment.")

        # Trapezoids use actual times and interpolated edges, rather than RHS sample counts.
        if has_gases:
            inside = (times > low) & (times < high)
            integration_times = np.r_[low, times[inside], high]
            for index, key in enumerate(self._GAS_KEYS):
                if not np.isfinite(trace[key]).all():
                    raise ValueError("The gas observation trace contains nonfinite values.")
                gas = np.interp(integration_times, times, trace[key])
                self._gas_integral[index] += np.sum(
                    np.diff(integration_times)*(gas[:-1] + gas[1:])/2.0)
            self._gas_seconds += high-low

        new = (times >= low-tolerance) & (times <= high+tolerance) & (times > self._last_time)
        new_times = times[new]
        if new_times.size:
            beat_ids = trace["time_since_beat_store"][new]
            hrs = trace["HR_store"][new]
            lvs = trace["V_lv_store"][new]
            rvs = trace["V_rv_store"][new]
            sas = trace["P_sa_store"][new]
            if (not all(np.isfinite(values).all() for values in (beat_ids, hrs, lvs, rvs, sas))
                    or np.any(hrs <= 0)):
                raise ValueError("The cardiac observation trace contains invalid values.")
            if np.any(beat_ids > new_times+tolerance):
                raise ValueError("A cardiac cycle starts after its observation time.")
            if np.any(np.diff(beat_ids) < 0):
                raise ValueError("Cardiac cycle identifiers moved backwards in time.")
            peak = float(np.max(hrs))
            self.peak_hr = peak if np.isnan(self.peak_hr) else max(self.peak_hr, peak)
            # Reduce thousands of RHS evaluations to a few cardiac-cycle summaries.
            starts = np.r_[0, np.flatnonzero(np.diff(beat_ids) != 0)+1]
            counts = np.diff(np.r_[starts, new_times.size])
            hr_sums = np.add.reduceat(hrs, starts)
            lv_maxima = np.maximum.reduceat(lvs, starts)
            lv_minima = np.minimum.reduceat(lvs, starts)
            rv_maxima = np.maximum.reduceat(rvs, starts)
            rv_minima = np.minimum.reduceat(rvs, starts)
            sa_maxima = np.maximum.reduceat(sas, starts)
            sa_minima = np.minimum.reduceat(sas, starts)
        else:
            starts = np.empty(0, dtype=int)
        for group, start in enumerate(starts):
            t = float(new_times[start])
            beat_id = float(beat_ids[start])
            if self._beat_id is None:
                # A first sample exactly at its beat boundary can begin a complete cycle.
                self._start_beat(beat_id, beat_id >= self.window_start-tolerance
                                 and t <= beat_id+tolerance)
            elif beat_id != self._beat_id:
                if beat_id < self._beat_id:
                    raise ValueError("Cardiac cycle identifiers moved backwards in time.")
                if self._beat_complete_start and self._beat_count and t <= self.window_end+tolerance:
                    self._completed.append([
                        self._beat_hr_sum/self._beat_count,
                        self._beat_lv_max, self._beat_lv_min,
                        self._beat_rv_max, self._beat_rv_min,
                        self._beat_sa_max, self._beat_sa_min,
                    ])
                self._start_beat(beat_id, beat_id >= self.window_start-tolerance)
            self._beat_lv_max = max(self._beat_lv_max, float(lv_maxima[group]))
            self._beat_lv_min = min(self._beat_lv_min, float(lv_minima[group]))
            self._beat_rv_max = max(self._beat_rv_max, float(rv_maxima[group]))
            self._beat_rv_min = min(self._beat_rv_min, float(rv_minima[group]))
            self._beat_sa_max = max(self._beat_sa_max, float(sa_maxima[group]))
            self._beat_sa_min = min(self._beat_sa_min, float(sa_minima[group]))
            self._beat_hr_sum += float(hr_sums[group])
            self._beat_count += int(counts[group])
        if new_times.size:
            self._last_time = float(new_times[-1])
        self._covered_until = high

    def summary(self):
        """Return HR (beats/s), LVEDV, LVESV, RVEDV, RVESV, systolic and diastolic P_sa for complete beats."""
        if self._covered_until < self.window_end-self._tolerance():
            raise ValueError("The observation window has not been fully covered.")
        if not self._completed:
            raise ValueError("The observation window contains no complete cardiac cycles.")
        return np.mean(np.asarray(self._completed, dtype=float), axis=0)

    def mean_gases(self):
        """Return physical-time means of arterial O2/CO2 tensions over the whole window."""
        duration = self.window_end-self.window_start
        if (self._covered_until < self.window_end-self._tolerance()
                or abs(self._gas_seconds-duration) > self._tolerance()):
            raise ValueError("Gas observations do not cover the whole observation window.")
        return self._gas_integral/duration
