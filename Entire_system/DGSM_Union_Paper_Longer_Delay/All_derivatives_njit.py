import numpy as np
import math
from Activation_Functions import activation_H, activation_H_derivative
from Resp_Control_Breath_Optimiser import calculate_single_V_dV_dt
from numba import njit

@njit
def accept_index_evals(finish_time, all_time, last_index, buffer_limit, i):
    if finish_time >= all_time[0]:  # No wrap-around
        idx_in_2 = np.searchsorted(all_time[:last_index + 1], finish_time, side='right')
    else:  # Wrap-around
        idx_in_sorted2 = np.searchsorted(all_time[last_index + 1:], finish_time, side='right')
        idx_in_2 = (idx_in_sorted2 + last_index + 1) % buffer_limit

    if idx_in_2 <= last_index:
        indices = np.arange(idx_in_2, last_index + 1)
    else:
        indices = np.concatenate((np.arange(idx_in_2, buffer_limit), np.arange(0, last_index + 1)))

    # Take every 3rd step, rk23  - optimized loop
    indices_len = len(indices)
    mask = np.array([(i - 2 - j) % 3 == 0 and (i - 2 - j) > 0 for j in range(indices_len)])
    accepted_index = [indices[indices_len - 1 - j] for j in range(indices_len) if mask[j]]

    return accepted_index


@njit
def get_delayed_value(t, delay, all_time, heart_index, buffer_limit, history_array, default_value):
    delay_time = t - delay

    if delay_time < 0:
        return default_value

    if delay_time >= all_time[0]:
        # No wrap-around
        delay_index1 = np.searchsorted(all_time[:heart_index + 1], delay_time, side='right')
    else:
        # Wrap-around
        idx_in_sorted = np.searchsorted(all_time[heart_index + 1:], delay_time, side='right')
        delay_index1 = (idx_in_sorted + heart_index + 1) % buffer_limit

    delay_index0 = (delay_index1 - 1) % buffer_limit
    t1 = all_time[delay_index1]
    t0 = all_time[delay_index0]
    v1 = history_array[delay_index1]
    v0 = history_array[delay_index0]

    if not (t0 <= delay_time <= t1):
        return default_value

    return float(v0 + (v1 - v0) * (delay_time - t0) / (t1 - t0))


@njit
def compute_mean_selected(HR_store, indices):
    total = 0.0
    for idx in indices:
        total += HR_store[idx]

    return total / len(indices)


@njit
def oxygen_content_from_po2(po2, pco2, C_O2_param1, C_O2_param2, C_O2_param3, scale_param3, scale_param4, PaCO2_n):
    po2_safe = max(po2, 1e-10)
    po2_virt = po2_safe * (PaCO2_n / max(pco2, 1e-10)) ** scale_param3
    so2 = (po2_virt ** C_O2_param2) / (po2_virt ** C_O2_param2 + scale_param4 ** C_O2_param2)
    return C_O2_param1 * 150 * so2 + C_O2_param3 * po2_safe


@njit
def invert_o2_content_to_po2(o2_content, pco2, C_O2_param1, C_O2_param2, C_O2_param3, scale_param3, scale_param4, PaCO2_n):
    lo = 1e-10
    hi = 1000.0

    if o2_content <= oxygen_content_from_po2(lo, pco2, C_O2_param1, C_O2_param2, C_O2_param3, scale_param3, scale_param4, PaCO2_n):
        return lo
    if o2_content >= oxygen_content_from_po2(hi, pco2, C_O2_param1, C_O2_param2, C_O2_param3, scale_param3, scale_param4, PaCO2_n):
        return hi

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        mid_content = oxygen_content_from_po2(mid, pco2, C_O2_param1, C_O2_param2, C_O2_param3, scale_param3, scale_param4, PaCO2_n)
        if mid_content < o2_content:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


@njit
def co2_content_from_pco2(pco2, po2, a2_gas, alpha2, beta2, C2, K2, Z):
    po2_safe = max(po2, 1e-10)
    pco2_safe = max(pco2, 1e-10)
    fco2 = max((pco2_safe * (1 + beta2 * po2_safe)) / (K2 * (1 + alpha2 * po2_safe)), 1e-10)
    fco2_root = fco2 ** (1 / a2_gas)
    return (C2 * Z) * fco2_root / (1 + fco2_root)


@njit
def invert_co2_content_to_pco2(co2_content, po2, a2_gas, alpha2, beta2, C2, K2, Z):
    po2_safe = max(po2, 1e-10)
    content_max = C2 * Z
    co2_content_safe = min(max(co2_content, 1e-10), content_max - 1e-10)
    co2_ratio = max(co2_content_safe / (content_max - co2_content_safe), 1e-10)
    return (co2_ratio ** a2_gas) * (K2 * (1 + alpha2 * po2_safe)) / (1 + beta2 * po2_safe)


@njit
def eval_spline(V, knots, coeffs):
    # clip to domain
    if V <= knots[0]:
        i = 0
    elif V >= knots[-1]:
        i = len(knots) - 2
    else:
        i = np.searchsorted(knots, V) - 1

    dx = V - knots[i]

    # coeffs[:, i] = [a, b, c, d]
    return (
        coeffs[0, i]*dx**3
        + coeffs[1, i]*dx**2
        + coeffs[2, i]*dx
        + coeffs[3, i]
    )


@njit(error_model="numpy")
def njit_compatible(t, state, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters, HR_store, time_since_beat_store,
    HR_every_store, Vu_ev_every_store, Vu_sv_every_store, Vu_rmv_every_store, Vu_amv_every_store, Emax_lv_every_store,
    Emax_rv_every_store, Vu_ev_store, Vu_sv_store, Vu_rmv_store, Vu_amv_store, Emax_lv_store, Emax_rv_store,
    f_sp_history, f_sh_history, f_v_history, f_sv_history, phi_met_history, Pa_O2_art_target_every_store, Pa_CO2_art_target_every_store,
    Nt_store, prev_flat_bit_store, finish_breath_time_store, Pa_O2_every_store, Pa_CO2_every_store, Pb_CO2_every_store,
    PamO2_store, PamCO2_store, PmbCO2_store, t1_store, t2_store, cs_t1, cs_t2, knots_1, knots_2,
    exercise_start_time):
    """
    Main derivative computation function with improved organization
    Computes all system derivatives in a single optimized function
    """
    # # State variables
    (  # Cardio state variables
        VT_pa, VT_pp, VT_pv, Q_pa,
        VT_la, VT_lv, VT_ra, VT_rv,
        VT_sv, VT_bv, VT_hv, VT_rmv, VT_amv, P_sp, P_sa, Q_sa, VT_vc,
        theta_ao, dtheta_ao_dt, theta_po, dtheta_po_dt, theta_mi, dtheta_mi_dt, theta_tr, dtheta_tr_dt,

        # Cardio controller state variables
        theta_change_O2_sp, theta_change_CO2_sp, theta_change_O2_sv, theta_change_CO2_sv, theta_change_O2_sh,
        theta_change_CO2_sh, P_tilda, f_ac, f_ap, R_ep_change, R_sp_change,
        R_rmp_n_change, R_amp_n_change, Vu_ev_change, Vu_sv_change, Vu_rmv_change, Vu_amv_change, Emax_lv_change,
        Emax_rv_change, Ts_change, Tv_change, xb_O2, xb_CO2, xh_O2, xh_CO2, Wh, xrm_O2, xrm_CO2, xam_O2, xM, x_met,
        P_n_current,

        # Gas exchange state variables
        Pd_1_O2, Pd_1_CO2, Pd_2_O2, Pd_2_CO2, Pd_3_O2, Pd_3_CO2, Pd_4_O2, Pd_4_CO2, Pd_5_O2, Pd_5_CO2, Pa_O2, Pa_CO2,
        dPa_O2_dt, dPa_CO2_dt, PA_O2, PA_CO2, PCSFCO2, MRTO2, MRTCO2, CTO2, CvtCO2, CBO2, CvbCO2, MRV,

        # Resp control state variable
        VE_integral
    ) = state

    # ============================================================================
    # PARAMETER EXTRACTION
    # ============================================================================
    (A_im, T_im, Tc, g_thor, P_thormax_n, P_thormin_n, VT_n, C_pa,
     C_pp, C_pv, L_pa, R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la, Emax_ra, P0_ra, KE_ra, C_sa,
     L_sa, R_sa, K1_vc, D1, Vvc_min, Kr_vc, Rvc_n, C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv,
     C_bv, C_hv, C_rmv, C_amv, kr_am, P_0, fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv,
     Io_v, kcc_sh, kcc_sp, kcc_sv, kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v,
     Wb_sh, Wb_sp, Wb_sv, Wc_sh, Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0,
     Emax_rv0, fes_min, GEmax_lv, GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0,
     R_rmp0, R_sp0, AT, g_ccsh, g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv, theta_shn, theta_spn,
     theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n, P_n_max, f_acCO2_n, f_ac_max, f_ac_min,
     k_ac, K_H, PaO2_ac_n, G_ap, DT_v, GT_s, GT_v, T0, A, B, C, D, Cvb_O2_n, gb_O2, R_bpn, Cvh_O2_n, Cvrm_O2_n, gh_O2,
     grm_O2, Kh_CO2, Krm_CO2, MO2_hpn, MO2_rmp, R_hpn, W_hn, Cvam_O2_n, gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max,
     phi_min, a2_gas, alpha2, beta2, C2, K2, PACO2_Delay_IC, PAO2_Delay_IC, P_atm, P_ws, Z, dc, KCCO2, MRBCO2, MO2_bp,
     MRTCO2_basal, MRTO2_basal, MRCO2, MRO2, s, GV_dead, KcCO2, KcMRV, KpCO2, KpO2, V0_dead, VA_rest, lambda1, lambda2,
     n, Pmax, Pmax_dot, E_rs, R_rs, P_ao,
     # added params
     Kp_ao, Kf_ao, Kb_ao, Kv_ao, theta_ao_max, Kp_mi, Kf_mi, Kb_mi, Kv_mi, theta_mi_max, Kp_po,
     Kf_po, Kb_po, Kv_po, theta_po_max, Kp_tr, Kf_tr, Kb_tr, Kv_tr, theta_tr_max, alpha_O2, R_po, R_mi, R_tr,
     R_ao, C_O2_param1, C_O2_param2, C_O2_param3, PAMO2_nominal,
     Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv, Vu_vc, Vu_pa, Vu_pp,
     Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev,
     tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_cc, tau_isc, tau_p, tau_z, tau_ac, tau_ap, tau_Ts, tau_Tv,
     tau_CO2, tau_O2, tau_w, tau_M, tau_met, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev, DV_rmv,
     DV_sv, DT_s, DT_v, Dmet, Fi_CO2, Fi_O2, Ta, T1, T2, VL_CO2, VL_O2, KCSFCO2, VB, tauMR, VTCO2, VTO2, tau_MRV,
     scale_param1, scale_param3, scale_param4, scale_param6,
     Pa_O2_lower, rise_time_atr, rise_time_ven,
     fall_time_ven, ahead1, theta_min, delta_P, r, l, V_nominal, V_scale
    ) = Input_Parameters

    # Determine the correct index based on t
    if t == 0:
        last_index = i % BUFFER_LIMIT
    elif i < 4 and num_removed == 3:
        last_index = i % BUFFER_LIMIT - 1
    else:
        last_index = (i - num_removed - 1) % BUFFER_LIMIT

    # ============================================================================
    # RESPIRATORY CONTROLLER
    # ============================================================================

    t1, t2 = t1_store[last_index], t2_store[last_index]

    finish_breath_time = finish_breath_time_store[last_index]

    time_since_last_breath = t - finish_breath_time

    if time_since_last_breath > (t1 + t2):
        accepted_indices = accept_index_evals(finish_breath_time, all_time, last_index, BUFFER_LIMIT, (i - num_removed))

        PamO2 = compute_mean_selected(Pa_O2_every_store, accepted_indices)
        PamCO2 = compute_mean_selected(Pa_CO2_every_store, accepted_indices)
        PmbCO2 = compute_mean_selected(Pb_CO2_every_store, accepted_indices)

    elif t != 0:
        PamO2 = PamO2_store[last_index]  # previous mean value
        PamCO2 = PamCO2_store[last_index]  # previous mean value
        PmbCO2 = PmbCO2_store[last_index]  # previous mean value

    else:
        PamO2 = Pa_O2
        PamCO2 = Pa_CO2

        PvbO2 = max(CBO2 / alpha_O2, 1)  # henry
        PvbCO2 = (max(CvbCO2 / (C2 * Z - CvbCO2), 1e-10) ** a2_gas) * (K2 * (1 + alpha2 * PvbO2)) / (1 + beta2 * PvbO2)
        G_bp = (1 / R_bpn) * (1 + xb_O2 + xb_CO2)
        R_bp = 1 / G_bp
        P_bv = (VT_bv - Vu_bv) / C_bv if VT_bv >= Vu_bv else VT_bv / C_bv
        Q_bp_1000 = max(((P_sp - P_bv) / R_bp), 2) / 1000
        Pb_CO2 = PvbCO2 + (PCSFCO2 - PvbCO2) * math.exp(-dc * (math.sqrt(Q_bp_1000 * KCCO2)))
        PmbCO2 = Pb_CO2

    G3 = KpO2 * ((PAMO2_nominal - PamO2) ** scale_param1) if PamO2 < PAMO2_nominal else 0
    VAflow = VA_rest * (KpCO2 * PamCO2 + KcCO2 * PmbCO2 + G3 + KcMRV * max(0, MRV) - (KpCO2 + KcCO2) * PaCO2_n)
    VAflow = min(max(VAflow, 0.04), 1.6)
    VD = GV_dead * VAflow + V0_dead

    if time_since_last_breath > (t1 + t2) or t == 0:
        finish_breath_time = finish_breath_time + (t1 + t2)  # update timestamp for the start of the next breath

        if t == 0:
            finish_breath_time = 0

        time_since_last_breath = t - finish_breath_time

        t1 = eval_spline(VAflow, knots_1, cs_t1)
        t2 = eval_spline(VAflow, knots_2, cs_t2)
    # ============================================================================
    # RESPIRATORY MECHANICS
    # ============================================================================

    # Respiratory timing constants
    BF = 1 / (t1 + t2)  # Breathing frequency
    TI = t1  # Inspiratory time
    VD_flow = BF * VD  # Dead space flow
    VE_flow = VAflow + VD_flow  # Total ventilation flow
    VT = VE_flow * (t1 + t2)  # Tidal volume

    # V = np.interp(time_since_last_breath, updates["current_times"], updates["V_current"])
    tolerance = 1e-4
    V, dV_dt = calculate_single_V_dV_dt(time_since_last_breath, (t1, t2), VAflow, VD, tolerance, E_rs, R_rs, P_ao)
    # P_musc, dP_musc_dt = calculate_single_P_musc_dP_dt(time_since_last_breath, (t1, t2), VAflow, VD, tolerance, E_rs, R_rs, P_ao)

    # for cardiovascular controller
    d_VE_integral_dt = VE_flow  # doesn't matter if this is VE_flow or 0 as NT only considers inspiration

    # ============================================================================
    # CARDIOVASCULAR CONTROLLER
    # ============================================================================
    T = 1 / HR_store[last_index]  # Heart period

    # Resistance calculations with improved organization
    R_ep = R_ep_change + R_ep0
    R_sp = R_sp_change + R_sp0

    # Active muscle resistance with metabolic feedback
    R_amp_n = R_amp_n_change + R_amp0
    R_amp = R_amp_n / (1 + xam_O2 + x_met)

    # Resting muscle resistance with CO2/O2 feedback
    R_rmp_n = R_rmp_n_change + R_rmp0
    R_rmp = R_rmp_n * (1 + xrm_CO2) / (1 + xrm_O2)

    G_bp = (1 / R_bpn) * (1 + xb_O2 + xb_CO2)
    R_bp = 1 / G_bp

    R_hp = R_hpn * (1 + xh_CO2) / (1 + xh_O2)

    # get the correct basal tissue CO2 production rate and exercise intensity from the inputs
    MRTCO2_basal = MRTCO2_basal - MRBCO2
    I = (MRTCO2 - MRTCO2_basal) / (AT - MRTCO2_basal)

    time_since_beat = time_since_beat_store[last_index]
    # Update after every heartbeat
    if t - time_since_beat > T:
        accepted_indices = accept_index_evals(time_since_beat, all_time, last_index, BUFFER_LIMIT, (i - num_removed))

        time_since_beat = time_since_beat + T

        HR = compute_mean_selected(HR_every_store, accepted_indices)
        T = 1 / HR
        Vu_ev = compute_mean_selected(Vu_ev_every_store, accepted_indices)
        Vu_sv = compute_mean_selected(Vu_sv_every_store, accepted_indices)
        Vu_rmv = compute_mean_selected(Vu_rmv_every_store, accepted_indices)
        Vu_amv = compute_mean_selected(Vu_amv_every_store, accepted_indices)
        Emax_lv = compute_mean_selected(Emax_lv_every_store, accepted_indices)
        Emax_rv = compute_mean_selected(Emax_rv_every_store, accepted_indices)


    elif t != 0:
        HR = HR_store[last_index]
        Vu_ev = Vu_ev_store[last_index]  # previous mean value
        Vu_sv = Vu_sv_store[last_index]  # previous mean value
        Vu_rmv = Vu_rmv_store[last_index]  # previous mean value
        Vu_amv = Vu_amv_store[last_index]  # previous mean value
        Emax_lv = Emax_lv_store[last_index]  # previous mean value
        Emax_rv = Emax_rv_store[last_index]  # previous mean value

    else:
        Vu_ev = max(Vu_ev_change + Vu_ev0, 0)
        Vu_sv = max(Vu_sv_change + Vu_sv0, 0)
        Vu_rmv = max(Vu_rmv_change + Vu_rmv0, 0)
        Vu_amv = max(Vu_amv_change + Vu_amv0, 0)
        Emax_lv = Emax_lv_change + Emax_lv0
        Emax_rv = Emax_rv_change + Emax_rv0

    # ============================================================================
    # CARDIOVASCULAR SYSTEM
    # ============================================================================
    # Muscle pump activation
    # alp ranges between 0 (beginning of muscle contraction) and 1
    # alp = (t % T_im) / T_im

    # # Muscle pump function
    # if (Tc / T_im) >= alp >= 0:
    #     psi = math.sin(np.pi * (T_im / Tc) * alp)
    # else:
    #     psi = 0
    #
    # P_im = A_im * psi  # Muscle pump pressure

    # p_im is 0 in resting conditions
    P_im = 0

    VT_change = VT - VT_n  # units of L
    T_resp = t1 + t2
    TE = t2
    # P_abdmax = P_abdmax_n + g_abd * VT_change
    P_thormax = P_thormax_n + g_thor * VT_change
    # P_abdmin = P_abdmin_n + g_abd * VT_change
    P_thormin = P_thormin_n + g_thor * VT_change

    S = time_since_last_breath / T_resp

    if 0 <= time_since_last_breath < TI:
        P_thor = P_thormax - (P_thormax - P_thormin) * (T_resp / TI) * S
        dP_thor_dt = -(P_thormax - P_thormin) / TI
    else:
        P_thor = P_thormax - (P_thormax - P_thormin) * ((TI + TE - T_resp * S) / TE)
        dP_thor_dt = (P_thormax - P_thormin) / TE

    # if 0 <= time_since_last_breath < (TI / 2):
    #     P_abd = P_abdmax - (P_abdmax - P_abdmin) * (T_resp / (TI / 2)) * S
    #
    # elif (TI / 2) <= time_since_last_breath < TI:
    #     P_abd = P_abdmin
    #
    # else:
    #     P_abd = P_abdmax - (P_abdmax - P_abdmin) * ((TI + TE - T_resp * S) / TE)

    # added P_thor to only the pulmonary compartments
    V_pa = (VT_pa - Vu_pa) * (VT_pa > Vu_pa)
    P_pa = V_pa / C_pa + P_thor  # 6-16mmHg

    V_pp = (VT_pp - Vu_pp) * (VT_pp > Vu_pp)
    P_pp = V_pp / C_pp + P_thor

    V_pv = (VT_pv - Vu_pv) * (VT_pv > Vu_pv)
    P_pv = V_pv / C_pv + P_thor

    ## The Heart
    # activation function for contraction of the ventricle and atria
    phi = activation_H(t - time_since_beat, 0, T, rise_time_atr, rise_time_ven, fall_time_ven, ahead1)
    phi_atr = activation_H(t - time_since_beat, 1, T, rise_time_atr, rise_time_ven, fall_time_ven, ahead1)
    dphi_dt = activation_H_derivative(t - time_since_beat, 0, T, rise_time_atr, rise_time_ven, fall_time_ven, ahead1)

    V_heart_peri = VT_la + VT_lv + VT_ra + VT_rv
    P_peri = math.exp(min((V_heart_peri - (Vu_ra + Vu_la + Vu_lv + Vu_rv + V_nominal)) / V_scale, 50))

    V_lv = (VT_lv - Vu_lv) * (VT_lv > Vu_lv)
    V_ra = (VT_ra - Vu_ra) * (VT_ra > Vu_ra)
    V_rv = (VT_rv - Vu_rv) * (VT_rv > Vu_rv)
    V_la = (VT_la - Vu_la) * (VT_la > Vu_la)

    P_lv = phi * Emax_lv * V_lv + (1 - phi) * P0_lv * (math.exp(KE_lv * V_lv) - 1) + P_thor + 1/l * P_peri
    P_ra = phi_atr * Emax_ra * V_ra + (1 - phi_atr) * P0_ra * (math.exp(KE_ra * V_ra) - 1) + P_thor + r * P_peri
    P_rv = phi * Emax_rv * V_rv + (1 - phi) * P0_rv * (math.exp(KE_rv * V_rv) - 1) + P_thor + 1/r * P_peri
    P_la = phi_atr * Emax_la * V_la + (1 - phi_atr) * P0_la * (math.exp(KE_la * V_la) - 1) + P_thor + l * P_peri


    # aortic valve
    valve_signal = 0.5 * (1 + np.tanh((P_lv - P_sa) / delta_P))
    if abs(valve_signal) < 1e-8:
        theta_ao = theta_min

    if theta_ao > theta_ao_max:
        theta_ao = theta_ao_max
    elif theta_ao < theta_min:
        theta_ao = theta_min

    # Compute area ratio with smooth transition
    AR_ao = ((1 - math.cos(theta_ao)) ** 2) / ((1 - math.cos(theta_ao_max)) ** 2)

    # Flow with smooth transition
    Q_lv = valve_signal * (math.sqrt(np.maximum(P_lv - P_sa, 0)) * AR_ao * R_ao)

    # Dynamics with smooth transition
    d2theta_ao_dt2 = valve_signal * ((P_lv - P_sa) * Kp_ao * math.cos(theta_ao) - Kf_ao * dtheta_ao_dt +
                                     Kb_ao * Q_lv * math.cos(theta_ao) - Kv_ao * Q_lv * math.sin(2 * theta_ao))

    ####################################

    valve_signal = 0.5 * (1 + np.tanh((P_la - P_lv) / delta_P))
    # Enforce theta bounds when nearly closed
    if abs(valve_signal) < 1e-8:
        theta_mi = theta_min  # minimum angle (closed)

    if theta_mi > theta_mi_max:
        theta_mi = theta_mi_max
    elif theta_mi < theta_min:
        theta_mi = theta_min

    # Compute area ratio with smooth transition
    AR_mi = ((1 - math.cos(theta_mi)) ** 2) / ((1 - math.cos(theta_mi_max)) ** 2)

    # Flow with smooth transition
    Q_mi = valve_signal * (math.sqrt(np.maximum(P_la - P_lv, 0)) * AR_mi * R_mi)

    # Dynamics with smooth transition
    d2theta_mi_dt2 = valve_signal * ((P_la - P_lv) * Kp_mi * math.cos(theta_mi) - Kf_mi * dtheta_mi_dt +
                                     Kb_mi * Q_mi * math.cos(theta_mi) - Kv_mi * Q_mi * math.sin(2 * theta_mi))

    ####################################
    valve_signal = 0.5 * (1 + np.tanh((P_rv - P_pa) / delta_P))

    # Enforce theta bounds when nearly closed
    if abs(valve_signal) < 1e-8:
        theta_po = theta_min  # minimum angle (closed)

    if theta_po > theta_po_max:
        theta_po = theta_po_max
        # AR_po = valve_signal * ((1 - math.cos(theta_po_max)) ** 2) / ((1 - math.cos(theta_po_max)) ** 2)
    elif theta_po < theta_min:
        theta_po = theta_min
        # AR_po = valve_signal * ((1 - math.cos(0.0872665)) ** 2) / ((1 - math.cos(theta_po_max)) ** 2)
    # else:
    #     AR_po = valve_signal * ((1 - math.cos(theta_po)) ** 2) / ((1 - math.cos(theta_po_max)) ** 2)

    # Compute area ratio with smooth transition
    AR_po = ((1 - math.cos(theta_po)) ** 2) / ((1 - math.cos(theta_po_max)) ** 2)

    # Flow with smooth transition
    Q_rv = valve_signal * (math.sqrt(np.maximum(P_rv - P_pa, 0)) * AR_po * R_po)

    # Dynamics with smooth transition
    d2theta_po_dt2 = valve_signal * ((P_rv - P_pa) * Kp_po * math.cos(theta_po) - Kf_po * dtheta_po_dt +
                                     Kb_po * Q_rv * math.cos(theta_po) - Kv_po * Q_rv * math.sin(2 * theta_po))

    ####################################
    valve_signal = 0.5 * (1 + np.tanh((P_ra - P_rv) / delta_P))

    # Enforce theta bounds when nearly closed
    if abs(valve_signal) < 1e-8:
        theta_tr = theta_min  # minimum angle (closed)

    if theta_tr > theta_tr_max:
        theta_tr = theta_tr_max
        # AR_tr = valve_signal * ((1 - math.cos(theta_tr_max)) ** 2) / ((1 - math.cos(theta_tr_max)) ** 2)
    elif theta_tr < theta_min:
        theta_tr = theta_min
        # AR_tr = valve_signal * ((1 - math.cos(0.0872665)) ** 2) / ((1 - math.cos(theta_tr_max)) ** 2)
    # else:
    #     AR_tr = valve_signal * ((1 - math.cos(theta_tr)) ** 2) / ((1 - math.cos(theta_tr_max)) ** 2)

    # # Compute area ratio with smooth transition
    AR_tr = ((1 - math.cos(theta_tr)) ** 2) / ((1 - math.cos(theta_tr_max)) ** 2)

    # Flow with smooth transition
    Q_tr = valve_signal * (math.sqrt(np.maximum(P_ra - P_rv, 0)) * AR_tr * R_tr)

    # Dynamics with smooth transition
    d2theta_tr_dt2 = valve_signal * ((P_ra - P_rv) * Kp_tr * math.cos(theta_tr) - Kf_tr * dtheta_tr_dt + Kb_tr * Q_tr *
                                     math.cos(theta_tr) - Kv_tr * Q_tr * math.sin(2 * theta_tr))

    ####################################

    Q_la = (P_pv - P_la) / R_pv
    Q_pp = max(((P_pp - P_pv) / R_pp), 0.0001)

    dVT_pa_dt = Q_rv - Q_pa
    dVT_pp_dt = Q_pa - Q_pp
    dVT_pv_dt = Q_pp - Q_la
    dQ_pa_dt = (P_pa - R_pa * Q_pa - P_pp) / L_pa

    dVT_lv_dt = Q_mi - Q_lv
    dVT_la_dt = Q_la - Q_mi

    dV_lv_dt = dVT_lv_dt * (VT_lv > Vu_lv)
    Wh_lv = (P_thor - P_lv) * dV_lv_dt

    if VT_vc >= Vu_vc:
        P_vc = D1 + K1_vc * (VT_vc - Vu_vc) + P_thor  # + source_values
        R_vc = Rvc_n
    else:
        K2_vc = (K1_vc * Vvc_min) / math.exp(Vu_vc / Vvc_min)  # for c1 continuity
        D2 = D1 - K2_vc * math.exp(Vu_vc / Vvc_min)  # for continuity
        P_vc = D2 + K2_vc * math.exp(max(VT_vc, Vvc_min) / Vvc_min) + P_thor  # + source_values
        R_vc = Kr_vc * (1 - (max(VT_vc, Vvc_min) / Vu_vc)) ** 2 + Rvc_n
    # C_vc = 1 / K1_vc
    # P_vc = V_vc / C_vc + P_thor

    Q_ra = (P_vc - P_ra) / R_vc

    dVT_rv_dt = Q_tr - Q_rv
    dVT_ra_dt = Q_ra - Q_tr

    dV_rv_dt = dVT_rv_dt * (VT_rv > Vu_rv)
    Wh_rv = (P_thor - P_rv) * dV_rv_dt

    ## systemic peripheral and venous circulation
    # splanchnic
    V_sv = (VT_sv - Vu_sv) * (VT_sv >= Vu_sv)
    P_sv = V_sv / C_sv

    Q_sp = (P_sp - P_sv) / R_sp

    # P_s = P_abd
    P_s = 0

    if P_vc < P_s:
        R_sv = R_sv_n * ((P_sv - P_vc) / (P_sv - P_s))
    else:
        R_sv = R_sv_n

    Q_sv = (P_sv - P_vc) / R_sv


    dVT_sv_dt = Q_sp - Q_sv

    # brain
    V_bv = (VT_bv - Vu_bv) * (VT_bv >= Vu_bv)
    P_bv = max(V_bv / C_bv, 0.0001)

    Q_bp = max((P_sp - P_bv) / R_bp, 2)

    P_b = 0

    if P_vc < P_b:
        R_bv = R_bv_n * ((P_bv - P_vc) / (P_bv - P_b))
    else:
        R_bv = R_bv_n


    Q_bv = (P_bv - P_vc) / R_bv

    dVT_bv_dt = Q_bp - Q_bv

    # coronary circulation
    V_hv = (VT_hv - Vu_hv) * (VT_hv >= Vu_hv)
    P_hv = max(V_hv / C_hv, 0.0001)

    Q_hp = max(((P_sp - P_hv) / R_hp), 0.0001)

    P_h = 0

    if P_vc < P_h:
        R_hv = R_hv_n * ((P_hv - P_vc) / (P_hv - P_h))
    else:
        R_hv = R_hv_n

    Q_hv = (P_hv - P_vc) / R_hv

    dVT_hv_dt = Q_hp - Q_hv

    # resting muscle
    # V_rmp = C_rmp * P_sp

    V_rmv = (VT_rmv - Vu_rmv) * (VT_rmv >= Vu_rmv)
    P_rmv = max(V_rmv / C_rmv, 0.0001)

    Q_rmp = max((P_sp - P_rmv) / R_rmp, 0.0001)

    P_rm = 0

    if P_vc < P_rm:
        R_rmv = R_rmv_n * ((P_rmv - P_vc) / (P_rmv - P_rm))
    else:
        R_rmv = R_rmv_n

    Q_rmv = (P_rmv - P_vc) / R_rmv

    dVT_rmv_dt = Q_rmp - Q_rmv

    # active muscle
    # V_amp = C_amp * P_sp

    P_0_am = Vu_amv / (C_amv * P_0)
    V_amv = (VT_amv - Vu_amv) * (VT_amv > Vu_amv)

    if VT_amv >= Vu_amv:
        P_amv = max(V_amv / C_amv + P_im, 0.0001)
    else:
        P_amv = max(P_im + P_0_am * (1 - (max(VT_amv, 0) / Vu_amv) ** -1.5), 0.0001)

    Q_amp = max((P_sp - P_amv), 0.0001) / R_amp

    P_am = 0

    if I > 0:
        R_amv = max(kr_am / VT_amv, 0.0001)
    elif P_vc < P_am:
        R_amv = R_amv_n * ((P_amv - P_vc) / (P_amv - P_am))
    else:
        R_amv = R_amv_n

    Q_amv = (P_amv - P_vc) / R_amv

    dVT_amv_dt = Q_amp - Q_amv

    ## systemic peripheral and venous circulation
    # extrasplanchnic
    # V_ep = C_ep * P_sp

    # C_jp = C_ep + C_sp + C_bp + C_hp + C_rmp + C_amp
    # Vu_jp = Vu_ep + Vu_sp + Vu_bp + Vu_hp + Vu_rmp + Vu_amp
    # Vu_jv = Vu_ev + Vu_sv + Vu_bv + Vu_hv + Vu_rmv + Vu_amv

    # V_u = Vu_sa + Vu_pa + Vu_pp + Vu_pv + Vu_ra + Vu_la + Vu_jp + Vu_jv + Vu_rv + Vu_lv + Vu_vc

    V_sa = P_sa * C_sa
    V_s_peripheral = P_sp * C_jp

    # left over volume
    # V_ev = (V_tot - V_sa - V_ra - V_rv - V_la - V_lv - V_pa - V_pp - V_pv - V_sv - V_rmv - V_amv - V_bv
    #         - V_hv - V_vc - V_u - V_s_peripheral)

    V_ev = max((V_tot - (V_sa + Vu_sa) - VT_ra - VT_rv - VT_la - VT_lv - VT_pa - VT_pp - VT_pv - VT_sv - VT_rmv - VT_amv -
            VT_bv - VT_hv - VT_vc - V_s_peripheral) - Vu_ev - Vu_jp, 1)

    P_ev = V_ev / C_ev  # + source_values

    Q_ep = (P_sp - P_ev) / R_ep

    P_e = 0

    if P_vc < P_e:
        R_ev = R_ev_n * ((P_ev - P_vc) / (P_ev - P_e))
    else:
        R_ev = R_ev_n

    Q_ev = (P_ev - P_vc) / R_ev

    Q_vc = Q_ev + Q_sv + Q_bv + Q_hv + Q_rmv + Q_amv
    Q_jp = Q_ep + Q_sp + Q_bp + Q_hp + Q_rmp + Q_amp

    # ignore VT_ev - doesn't add to the equations
    dP_sa_dt = (Q_lv - Q_sa) / C_sa
    # dVT_ev_dt = Q_ep - Q_ev
    dVT_vc_dt = Q_vc - Q_ra
    dP_sp_dt = (Q_sa - Q_jp) / C_jp
    dQ_sa_dt = (P_sa - P_thor - R_sa * Q_sa - P_sp) / L_sa

    # VT_sa = V_sa + Vu_sa
    # should be + ?, edit: removed P_thor from here. Ignore

    # AA = (VT_lv + VT_rv + VT_la + VT_ra + (V_sa + Vu_sa) + VT_amv + VT_rmv + (V_ev + Vu_ev) + VT_sv + VT_hv + VT_bv +
    #       (V_s_peripheral + Vu_jp) + VT_vc + VT_pa + VT_pp + VT_pv)

    # ============================================================================
    # GAS EXCHANGE
    # ============================================================================
    # Dead space gas exchange rate
    constant = (abs(dV_dt) / (0.2 * VD))

    if dV_dt >= 0:  # Inspiration
        # Inspired gas partial pressures
        PiO2 = Fi_O2 * (P_atm - P_ws) / 100
        PiCO2 = Fi_CO2 * (P_atm - P_ws) / 100

        # Dead space gas exchange during inspiration
        dPd_1_O2_dt = constant * (PiO2 - Pd_1_O2)
        dPd_1_CO2_dt = constant * (PiCO2 - Pd_1_CO2)

        dPd_2_O2_dt = constant * (Pd_1_O2 - Pd_2_O2)
        dPd_2_CO2_dt = constant * (Pd_1_CO2 - Pd_2_CO2)

        dPd_3_O2_dt = constant * (Pd_2_O2 - Pd_3_O2)
        dPd_3_CO2_dt = constant * (Pd_2_CO2 - Pd_3_CO2)

        dPd_4_O2_dt = constant * (Pd_3_O2 - Pd_4_O2)
        dPd_4_CO2_dt = constant * (Pd_3_CO2 - Pd_4_CO2)

        dPd_5_O2_dt = constant * (Pd_4_O2 - Pd_5_O2)  # edited to just have one deadspace
        dPd_5_CO2_dt = constant * (Pd_4_CO2 - Pd_5_CO2)
    else:
        dPd_1_O2_dt = constant * (Pd_2_O2 - Pd_1_O2)
        dPd_1_CO2_dt = constant * (Pd_2_CO2 - Pd_1_CO2)

        dPd_2_O2_dt = constant * (Pd_3_O2 - Pd_2_O2)
        dPd_2_CO2_dt = constant * (Pd_3_CO2 - Pd_2_CO2)

        dPd_3_O2_dt = constant * (Pd_4_O2 - Pd_3_O2)
        dPd_3_CO2_dt = constant * (Pd_4_CO2 - Pd_3_CO2)

        dPd_4_O2_dt = constant * (Pd_5_O2 - Pd_4_O2)
        dPd_4_CO2_dt = constant * (Pd_5_CO2 - Pd_4_CO2)

        dPd_5_O2_dt = constant * (PA_O2 - Pd_5_O2)
        dPd_5_CO2_dt = constant * (PA_CO2 - Pd_5_CO2)

    # Ta = LCTV / Q_la
    # Ta = 6  # decreased to have a smaller circular buffer

    CeCO2 = co2_content_from_pco2(PA_CO2, PA_O2, a2_gas, alpha2, beta2, C2, K2, Z)

    # alpha_O2 = 0.0000317
    # alpha_CO2 = 0.000667

    # FO2 = (PA_O2 * (1 + beta1 * PA_CO2)) / (K1 * (1 + alpha1 * PA_CO2))
    PAO2_virt = max(PA_O2 * (PaCO2_n / max(PA_CO2, 1e-10)) ** scale_param3, 1e-10)
    SaO2 = (PAO2_virt ** C_O2_param2) / (PAO2_virt ** C_O2_param2 + scale_param4 ** C_O2_param2)
    CeO2 = (C_O2_param1 * 150 * SaO2) + C_O2_param3 * PA_O2

    # Gas transport
    # Brain
    MRBO2 = MO2_bp / 1000

    # Body Tissues Compartment
    MRTO2_basal = MRTO2_basal - MRBO2

    MRCO2 = MRCO2 - MRBCO2
    MRO2 = MRO2 - MRBO2

    # exercise
    # if 100 < t <= 600:
    #     MRCO2 = 0.4 / 60 - MRBCO2
    #     MRO2 = 0.45 / 60 - MRBO2

    #
    # if 500 < t <= 700:
    #     MRCO2 = 0.6 / 60 - MRBCO2
    #     MRO2 = 0.65 / 60 - MRBO2
    #
    if t >= exercise_start_time:
        MRCO2 = 1.2 / 60 - MRBCO2
        MRO2 = 1.2 / 60 - MRBO2

    # if 210 < t:
    #     MRCO2 = 1 / 60 - MRBCO2
    #     MRO2 = 1.05 / 60 - MRBO2

    ## new code
    # PvbCO2 and PvbO2 is the same as the brain compartment CO2 and O2 partial pressure
    # CvbO2 is NOT the same as CBO2 (CBO2 doesn't include haemoglobin), but here CvbCO2 is the SAME as CBCO2 (just the curve)

    # brain
    PvbO2 = max(CBO2 / alpha_O2, 1)  # henry
    CvbCO2_ratio = max(CvbCO2 / (C2 * Z - CvbCO2), 1e-10)
    PvbCO2 = (CvbCO2_ratio ** a2_gas) * (K2 * (1 + alpha2 * PvbO2)) / (
            1 + beta2 * PvbO2)  # haldane effect/ CO2 dissociation curve

    # FbO2 = (PvbO2 * (1 + beta1 * PvbCO2)) / (K1 * (1 + alpha1 * PvbCO2))  # bohr curve
    # CvbO2_1 = (C1 * Z) * (FbO2 ** (1 / a1)) / (1 + (FbO2 ** (1 / a1)))  # bohr curve

    PvbO2_virt = PvbO2 * (PaCO2_n / max(PvbCO2, 1e-10)) ** scale_param3
    SvbO2 = (PvbO2_virt ** C_O2_param2) / (PvbO2_virt ** C_O2_param2 + scale_param4 ** C_O2_param2)
    CvbO2 = C_O2_param1 * 150 * SvbO2 + C_O2_param3 * PvbO2

    # tissue
    PvtO2 = max(CTO2 / alpha_O2, 1)  # henry
    # if CTO2 slightly negative but goes back later, it's fine. Not fine if it decreases below -1
    # if CTO2 < -1:
    #     PvtO2 = CTO2 / alpha_O2
    CvtCO2_ratio = max(CvtCO2 / (C2 * Z - CvtCO2), 1e-10)
    PvtCO2 = (CvtCO2_ratio ** a2_gas) * (K2 * (1 + alpha2 * PvtO2)) / (
            1 + beta2 * PvtO2)  # haldane effect/ CO2 dissociation curve

    # serna and carlos
    # FtO2 = (PvtO2 * (1 + beta1 * PvtCO2)) / (K1 * (1 + alpha1 * PvtCO2))  # bohr curve
    # CvtO2_1 = (C1 * Z) * (FtO2 ** (1 / a1)) / (1 + (FtO2 ** (1 / a1)))  # bohr curve
    # ursino model 1997
    PvtO2_virt = PvtO2 * (PaCO2_n / max(PvtCO2, 1e-10)) ** scale_param3
    SvtO2 = (PvtO2_virt ** C_O2_param2) / (PvtO2_virt ** C_O2_param2 + scale_param4 ** C_O2_param2)
    CvtO2 = C_O2_param1 * 150 * SvtO2 + C_O2_param3 * PvtO2

    Q_bp_1000 = Q_bp / 1000
    Q_pp_1000 = max(Q_pp / 1000, Q_bp_1000)

    QT = max(Q_pp_1000 - Q_bp_1000, 0.0001)

    # overall CvO2 and CvCO2
    CvO2 = (Q_bp_1000 / Q_pp_1000) * CvbO2 + (QT / Q_pp_1000) * CvtO2
    CvCO2 = (Q_bp_1000 / Q_pp_1000) * CvbCO2 + (QT / Q_pp_1000) * CvtCO2

    CaO2 = (1 - s) * CeO2 + s * CvO2
    CaCO2 = (1 - s) * CeCO2 + s * CvCO2

    Pa_O2_art_target_every = invert_o2_content_to_po2(
        CaO2, Pa_CO2, C_O2_param1, C_O2_param2, C_O2_param3, scale_param3, scale_param4, PaCO2_n
    )
    Pa_O2_delay = get_delayed_value(t, Ta, all_time, last_index, BUFFER_LIMIT, Pa_O2_art_target_every_store, Pa_O2_art_target_every)
    d2Pa_O2_dt2 = (Pa_O2_delay - (T1 + T2) * dPa_O2_dt - Pa_O2) / (T1 * T2)

    Pa_CO2_art_target_every = invert_co2_content_to_pco2(
        CaCO2, Pa_O2_art_target_every, a2_gas, alpha2, beta2, C2, K2, Z
    )
    Pa_CO2_delay = get_delayed_value(t, Ta, all_time, last_index, BUFFER_LIMIT, Pa_CO2_art_target_every_store, Pa_CO2_art_target_every)
    d2Pa_CO2_dt2 = (Pa_CO2_delay - (T1 + T2) * dPa_CO2_dt - Pa_CO2) / (T1 * T2)

    dCBO2_dt = (-MRBO2 + Q_bp_1000 * (CaO2 - CvbO2)) / VB  # brain volume for conc is 0.9
    dCvbCO2_dt = (MRBCO2 + Q_bp_1000 * (CaCO2 - CvbCO2)) / VB  # brain volume for conc is 0.9

    dCTO2_dt = (-MRTO2 + QT * (CaO2 - CvtO2)) / VTO2
    dCvtCO2_dt = (MRTCO2 + QT * (CaCO2 - CvtCO2)) / VTCO2

    Pb_CO2 = PvbCO2 + (PCSFCO2 - PvbCO2) * math.exp(-dc * (math.sqrt(Q_bp_1000 * KCCO2)))
    # Pb_CO2 = 43
    # dPvbCO2_dt = (MRBCO2 + Q_pp_1000 * SCO2 * (Pa_CO2 - PvbCO2) - h) / SbCO2
    dPCSFCO2_dt = (PvbCO2 - PCSFCO2) / KCSFCO2

    dMRTO2_dt = (MRO2 - MRTO2) / tauMR
    dMRTCO2_dt = (MRCO2 - MRTCO2) / tauMR

    # cO2_diff = QT * (CaO2 - CvtO2)
    # cCO2_diff = QT * (CaCO2 - CvtCO2)

    V_O2 = VL_O2  # removed + V as this helps decrease VAflow (decreased time constant for ventilation)
    V_CO2 = VL_CO2

    if dV_dt >= 0:  # deadspace PAO2 is increasing towards 150
        dPA_O2_dt = (863 * Q_pp_1000 * (CvO2 - CeO2) * (1 - s) + dV_dt * (
                Pd_5_O2 - PA_O2)) / V_O2  # 863 is unit conversion. First from stpd to btps (x 1.21), then into pressure (x 713, P_atm - P_h20)
        dPA_CO2_dt = (863 * Q_pp_1000 * (CvCO2 - CeCO2) * (1 - s) + dV_dt * (Pd_5_CO2 - PA_CO2)) / V_CO2

    else:  # deadspace PAO2 is decreasing towards PA_O2 during expiration
        dPA_O2_dt = (863 * Q_pp_1000 * (CvO2 - CeO2) * (1 - s)) / V_O2
        dPA_CO2_dt = (863 * Q_pp_1000 * (CvCO2 - CeCO2) * (1 - s)) / V_CO2

    # Metabolism Dynamic
    MRR = max((MRBCO2 + MRBO2 + MRTCO2 + MRTO2) / (MRBCO2 + MRBO2 + MRTCO2_basal + MRTO2_basal), 1)
    # MRV = 0 if MRV < 0 or MRR <= 1 else MRV
    dMRV_dt = ((MRR - 1) - MRV) / tau_MRV

    # # Cardiovascular Controller

    if time_since_last_breath < t1:
        prev_flat_bit = prev_flat_bit_store[last_index]
        Nt = VE_integral - prev_flat_bit  # Take value minus previous flat bit
    else:
        Nt = Nt_store[last_index] * math.exp(-(t - all_time[last_index]) / (t2 / math.log(1000)))
        prev_flat_bit = VE_integral

    ## CNS Ischemic Response
    w_sp = x_sp / (1 + math.exp((Pa_O2 - PO2_sp) / kisc_sp))
    theta_sp = theta_spn - theta_change_O2_sp - theta_change_CO2_sp
    dtheta_change_O2_sp_dt = (-theta_change_O2_sp + w_sp) / tau_isc
    dtheta_change_CO2_sp_dt = (-theta_change_CO2_sp + g_ccsp * (Pa_CO2 - PaCO2_n)) / tau_cc

    w_sv = x_sv / (1 + math.exp((Pa_O2 - PO2_sv) / kisc_sv))
    theta_sv = theta_svn - theta_change_O2_sv - theta_change_CO2_sv
    dtheta_change_O2_sv_dt = (-theta_change_O2_sv + w_sv) / tau_isc
    dtheta_change_CO2_sv_dt = (-theta_change_CO2_sv + g_ccsv * (Pa_CO2 - PaCO2_n)) / tau_cc

    w_sh = x_sh / (1 + math.exp((Pa_O2 - PO2_sh) / kisc_sh))
    theta_sh = theta_shn - theta_change_O2_sh - theta_change_CO2_sh
    dtheta_change_O2_sh_dt = (-theta_change_O2_sh + w_sh) / tau_isc
    dtheta_change_CO2_sh_dt = (-theta_change_CO2_sh + g_ccsh * (Pa_CO2 - PaCO2_n)) / tau_cc

    ## Afferent Pathways
    # exp_arg = np.clip((P_tilda - P_n) / k_ab, -40, 40)  # Prevent overflow
    P_n_resultant = P_n + I * (P_n_max - P_n)
    dP_n_current_dt = (-P_n_current + P_n_resultant) / tau_p

    exp_arg = (P_tilda - P_n_current) / k_ab
    f_ab = (f_ab_min + f_ab_max * math.exp(exp_arg)) / (1 + math.exp(exp_arg))
    dP_tilda_dt = (P_sa + tau_z * dP_sa_dt - P_tilda) / tau_p

    # afferent chemoreflex pathway constant parameters
    if Pa_O2 >= Pa_O2_lower:
        K = K_H
    else:
        K = K_H - (scale_param6 * (Pa_O2 - Pa_O2_lower))
    # else:
    #     K = K_H - scale_param8

    phi_ac = ((f_ac_max + f_ac_min * math.exp((Pa_O2 - PaO2_ac_n) / k_ac)) / (1 + math.exp((Pa_O2 - PaO2_ac_n) / k_ac)) *
              (K * math.log(Pa_CO2 / PaCO2_n) + f_acCO2_n))

    d_fac_dt = (phi_ac - f_ac) / tau_ac

    # afferent activity from Pulmonary Stretch Receptors constant parameters
    phi_ap = G_ap * VT
    df_ap_dt = (phi_ap - f_ap) / tau_ap

    ## Efferent Pathways constant parameters
    Y_sh = (Ysh_min + Ysh_max * math.exp((I - Io_sh) / kcc_sh)) / (1 + math.exp((I - Io_sh) / kcc_sh))
    f_ash = Wt_sh * Nt + Wb_sh * f_ab + Wc_sh * f_ac + Wp_sh * f_ap - theta_sh
    f_sh = min(fes_max, (fes_inf + (fes_o - fes_inf) * math.exp(kes * f_ash) + Y_sh))

    Y_sp = (Ysp_min + Ysp_max * math.exp((I - Io_sp) / kcc_sp)) / (1 + math.exp((I - Io_sp) / kcc_sp))
    f_asp = Wt_sp * Nt + Wb_sp * f_ab + Wc_sp * f_ac + Wp_sp * f_ap - theta_sp
    f_sp = min(fes_max, (fes_inf + (fes_o - fes_inf) * math.exp(kes * f_asp) + Y_sp))

    Y_sv = (Ysv_min + Ysv_max * math.exp((I - Io_sv) / kcc_sv)) / (1 + math.exp((I - Io_sv) / kcc_sv))
    f_asv = Wt_sv * Nt + Wb_sv * f_ab + Wc_sv * f_ac + Wp_sv * f_ap - theta_sv
    f_sv = min(fes_max, (fes_inf + (fes_o - fes_inf) * math.exp(kes * f_asv) + Y_sv))

    Y_v = (Yv_min + Yv_max * math.exp((I - Io_v) / kcc_v)) / (1 + math.exp((I - Io_v) / kcc_v))
    first_term = (fev_o + fev_inf * math.exp((f_ab - fab_o) / kev)) / (1 + math.exp((f_ab - fab_o) / kev))
    f_v = first_term - Wt_v * Nt + Wc_v * f_ac + Wp_v * f_ap - theta_v + Y_v # changed
    f_v = max(f_v, 0)


    # Fetch delayed values
    f_sp_delay2_Ramp = get_delayed_value(t, DR_amp, all_time, last_index, BUFFER_LIMIT, f_sp_history, f_sp)
    f_sp_delay2_Rep = get_delayed_value(t, DR_ep, all_time, last_index, BUFFER_LIMIT, f_sp_history, f_sp)
    f_sp_delay2_Rrmp = get_delayed_value(t, DR_rmp, all_time, last_index, BUFFER_LIMIT, f_sp_history, f_sp)
    f_sp_delay2_Rsp = get_delayed_value(t, DR_sp, all_time, last_index, BUFFER_LIMIT, f_sp_history, f_sp)

    f_sv_delay5_Vu_ev = get_delayed_value(t, DV_ev, all_time, last_index, BUFFER_LIMIT, f_sv_history, f_sv)
    f_sv_delay5_Vu_sv = get_delayed_value(t, DV_sv, all_time, last_index, BUFFER_LIMIT, f_sv_history, f_sv)
    f_sv_delay5_Vu_rmv = get_delayed_value(t, DV_rmv, all_time, last_index, BUFFER_LIMIT, f_sv_history, f_sv)
    f_sv_delay5_Vu_amv = get_delayed_value(t, DV_amv, all_time, last_index, BUFFER_LIMIT, f_sv_history, f_sv)

    f_sh_delay2_Emax_lv = get_delayed_value(t, DEmax_lv, all_time, last_index, BUFFER_LIMIT, f_sh_history, f_sh)
    f_sh_delay2_Emax_rv = get_delayed_value(t, DEmax_rv, all_time, last_index, BUFFER_LIMIT, f_sh_history, f_sh)

    f_sh_delay2_s = get_delayed_value(t, DT_s, all_time, last_index, BUFFER_LIMIT, f_sh_history, f_sh)
    f_v_delay0_2 = get_delayed_value(t, DT_v, all_time, last_index, BUFFER_LIMIT, f_v_history, f_v)

    # heart period
    sigma_Ts = GT_s * math.log(max(f_sh_delay2_s, fes_min) - fes_min + 1)
    d_Ts_change_dt = (- Ts_change + sigma_Ts) / tau_Ts

    sigma_Tv = GT_v * f_v_delay0_2
    d_Tv_change_dt = (- Tv_change + sigma_Tv) / tau_Tv

    T = max(Ts_change + Tv_change + T0, 0.25)  # prevent max HR > 240 bpm
    HR_every = 1 / T

    if t == 0:
        HR = HR_every


    # continue with equations
    sigma_Rep = GR_ep * math.log(max(f_sp_delay2_Rep, fes_min) - fes_min + 1)
    sigma_Rsp = GR_sp * math.log(max(f_sp_delay2_Rsp, fes_min) - fes_min + 1)
    sigma_Rrmp_n = GR_rmp * math.log(max(f_sp_delay2_Rrmp, fes_min) - fes_min + 1)
    sigma_Ramp_n = GR_amp * math.log(max(f_sp_delay2_Ramp, fes_min) - fes_min + 1)

    sigma_Vu_ev = GV_ev * math.log(max(f_sv_delay5_Vu_ev, fes_min) - fes_min + 1)
    sigma_Vu_sv = GV_sv * math.log(max(f_sv_delay5_Vu_sv, fes_min) - fes_min + 1)
    sigma_Vu_rmv = GV_rmv * math.log(max(f_sv_delay5_Vu_rmv, fes_min) - fes_min + 1)
    sigma_Vu_amv = GV_amv * math.log(max(f_sv_delay5_Vu_amv, fes_min) - fes_min + 1)

    sigma_Emax_lv = GEmax_lv * math.log(max(f_sh_delay2_Emax_lv, fes_min) - fes_min + 1)
    sigma_Emax_rv = GEmax_rv * math.log(max(f_sh_delay2_Emax_rv, fes_min) - fes_min + 1)

    dR_ep_change_dt = (- R_ep_change + sigma_Rep) / tau_Rep
    dR_sp_change_dt = (- R_sp_change + sigma_Rsp) / tau_Rsp
    dR_rmp_n_change_dt = (- R_rmp_n_change + sigma_Rrmp_n) / tau_Rrmp
    dR_amp_n_change_dt = (- R_amp_n_change + sigma_Ramp_n) / tau_Ramp

    dVu_ev_change_dt = (- Vu_ev_change + sigma_Vu_ev) / tau_Vev
    dVu_sv_change_dt = (- Vu_sv_change + sigma_Vu_sv) / tau_Vsv
    dVu_rmv_change_dt = (- Vu_rmv_change + sigma_Vu_rmv) / tau_Vrmv
    dVu_amv_change_dt = (- Vu_amv_change + sigma_Vu_amv) / tau_Vamv
    Vu_ev_every = max(Vu_ev_change + Vu_ev0, 0)
    Vu_sv_every = max(Vu_sv_change + Vu_sv0, 0)
    Vu_rmv_every = max(Vu_rmv_change + Vu_rmv0, 0)
    Vu_amv_every = max(Vu_amv_change + Vu_amv0, 0)

    dEmax_lv_change_dt = (- Emax_lv_change + sigma_Emax_lv) / tau_Emax_lv
    dEmax_rv_change_dt = (- Emax_rv_change + sigma_Emax_rv) / tau_Emax_rv
    Emax_lv_every = Emax_lv_change + Emax_lv0
    Emax_rv_every = Emax_rv_change + Emax_rv0

    ## Blood Flow Local Control
    # Cvb_O2 = CaO2 - MO2_bp / Q_bp
    dxb_O2_dt = (- xb_O2 - gb_O2 * (CvbO2 - Cvb_O2_n)) / tau_O2

    numerator = A + (B / (1 + C * math.exp(D * math.log10(Pa_CO2))))
    denominator = A + (B / (1 + C * math.exp(D * math.log10(PaCO2_n))))
    phi_b = numerator / denominator - 1
    dxb_CO2_dt = (- xb_CO2 - phi_b) / tau_CO2

    # coronary
    MO2_hp = MO2_hpn * Wh / W_hn
    Cvh_O2 = CaO2 - MO2_hp / Q_hp
    dxh_O2_dt = (- xh_O2 - gh_O2 * (Cvh_O2 - Cvh_O2_n)) / tau_O2

    phi_h = (1 - math.exp((Pa_CO2 - PaCO2_n) / Kh_CO2)) / (1 + math.exp((Pa_CO2 - PaCO2_n) / Kh_CO2))
    dxh_CO2_dt = (- xh_CO2 + phi_h) / tau_CO2

    wh = Wh_lv + Wh_rv
    dWh_dt = (wh - Wh) / tau_w

    # resting muscle
    Cvrm_O2 = CaO2 - MO2_rmp / Q_rmp
    dxrm_O2_dt = (- xrm_O2 - grm_O2 * (Cvrm_O2 - Cvrm_O2_n)) / tau_O2

    phi_rm = (1 - math.exp((Pa_CO2 - PaCO2_n) / Krm_CO2)) / (1 + math.exp((Pa_CO2 - PaCO2_n) / Krm_CO2))
    dxrm_CO2_dt = (- xrm_CO2 + phi_rm) / tau_CO2

    # active muscle blood flow
    MO2_amp = MO2_ampn * (1 + xM)
    Cvam_O2 = CaO2 - MO2_amp / Q_amp
    dxam_O2_dt = (- xam_O2 - gam_O2 * (Cvam_O2 - Cvam_O2_n)) / tau_O2

    dxM_dt = (- xM + gM * I) / tau_M

    phi_met = (phi_min + phi_max * math.exp((I - Io_met) / kmet)) / (1 + math.exp((I - Io_met) / kmet))
    phi_met_delay = get_delayed_value(t, Dmet, all_time, last_index, BUFFER_LIMIT, phi_met_history, phi_met)
    # phi_met_delay = phi_met

    dx_met_dt = (- x_met + phi_met_delay) / tau_met
    dP_peri_dt = P_peri * (dVT_la_dt + dVT_lv_dt + dVT_ra_dt + dVT_rv_dt) / V_scale
    dP_rv_dt = Emax_rv * (phi * dV_rv_dt + dphi_dt*(VT_rv - Vu_rv)) + P0_rv * (-dphi_dt * (math.exp(KE_rv * (VT_rv - Vu_rv)) - 1) + (1 - phi) * KE_rv * math.exp(KE_rv * (VT_rv - Vu_rv)) * dVT_rv_dt) + dP_thor_dt + 1/r * dP_peri_dt
    dP_lv_dt = Emax_lv * (phi * dV_lv_dt + dphi_dt*(VT_lv - Vu_lv)) + P0_lv * (-dphi_dt * (math.exp(KE_lv * (VT_lv - Vu_lv)) - 1) + (1 - phi) * KE_lv * math.exp(KE_lv * (VT_lv - Vu_lv)) * dVT_lv_dt) + dP_thor_dt + 1/l * dP_peri_dt

    # ============================================================================
    # RETURN ALL COMPUTED VALUES
    # ============================================================================
    return (time_since_beat,
            HR, Vu_ev, Vu_sv, Vu_rmv, Vu_amv,
            Emax_lv, Emax_rv, f_sp, f_sh, f_v, f_sv, phi_met, HR_every, Vu_ev_every, Vu_sv_every,
            Vu_rmv_every, Vu_amv_every, Emax_lv_every, Emax_rv_every,
            prev_flat_bit,

            # for targets
            VT_lv, VT_rv, P_sa, P_rv, P_la, VT_la, VT_ra, P_ra, P_lv, phi, phi_atr, V, VAflow, Q_pp, theta_ao, theta_po, theta_mi, theta_tr, dP_rv_dt, dP_lv_dt, P_peri,

            # Gas exchange outputs
            Pa_O2, Pa_CO2, Pb_CO2,
            PA_O2, PA_CO2, Nt,
            Pa_O2_art_target_every, Pa_CO2_art_target_every,

            t1, t2, finish_breath_time, PamO2, PamCO2, PmbCO2,

            dVT_pa_dt, dVT_pp_dt, dVT_pv_dt, dQ_pa_dt, dVT_la_dt, dVT_lv_dt, dVT_ra_dt, dVT_rv_dt, dVT_sv_dt,
            dVT_bv_dt, dVT_hv_dt, dVT_rmv_dt, dVT_amv_dt, dP_sp_dt, dP_sa_dt, dQ_sa_dt, dVT_vc_dt,
            dtheta_ao_dt, d2theta_ao_dt2, dtheta_po_dt, d2theta_po_dt2, dtheta_mi_dt, d2theta_mi_dt2, dtheta_tr_dt,
            d2theta_tr_dt2,

            # cardio controller derivatives
            dtheta_change_O2_sp_dt, dtheta_change_CO2_sp_dt, dtheta_change_O2_sv_dt, dtheta_change_CO2_sv_dt,
            dtheta_change_O2_sh_dt, dtheta_change_CO2_sh_dt, dP_tilda_dt, d_fac_dt, df_ap_dt, dR_ep_change_dt,
            dR_sp_change_dt, dR_rmp_n_change_dt, dR_amp_n_change_dt, dVu_ev_change_dt, dVu_sv_change_dt,
            dVu_rmv_change_dt, dVu_amv_change_dt, dEmax_lv_change_dt, dEmax_rv_change_dt, d_Ts_change_dt,
            d_Tv_change_dt, dxb_O2_dt, dxb_CO2_dt, dxh_O2_dt, dxh_CO2_dt, dWh_dt, dxrm_O2_dt, dxrm_CO2_dt, dxam_O2_dt,
            dxM_dt, dx_met_dt, dP_n_current_dt,

            # gas exchange derivatives
            dPd_1_O2_dt, dPd_1_CO2_dt, dPd_2_O2_dt, dPd_2_CO2_dt, dPd_3_O2_dt, dPd_3_CO2_dt, dPd_4_O2_dt,
            dPd_4_CO2_dt, dPd_5_O2_dt, dPd_5_CO2_dt, dPa_O2_dt, dPa_CO2_dt, d2Pa_O2_dt2, d2Pa_CO2_dt2, dPA_O2_dt,
            dPA_CO2_dt, dPCSFCO2_dt, dMRTO2_dt, dMRTCO2_dt, dCTO2_dt, dCvtCO2_dt, dCBO2_dt, dCvbCO2_dt, dMRV_dt,

            # resp control derivatives
            d_VE_integral_dt)


@njit(error_model="numpy")
def model_derivatives_njit(
    t, state, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters,
    HR_store, time_since_beat_store, HR_every_store, Vu_ev_every_store,
    Vu_sv_every_store, Vu_rmv_every_store, Vu_amv_every_store,
    Emax_lv_every_store, Emax_rv_every_store, Vu_ev_store, Vu_sv_store,
    Vu_rmv_store, Vu_amv_store, Emax_lv_store, Emax_rv_store, f_sp_history,
    f_sh_history, f_v_history, f_sv_history, phi_met_history,
    Pa_O2_art_target_every_store, Pa_CO2_art_target_every_store, Nt_store,
    prev_flat_bit_store, finish_breath_time_store, Pa_O2_every_store,
    Pa_CO2_every_store, Pb_CO2_every_store, PamO2_store, PamCO2_store,
    PmbCO2_store, t1_store, t2_store, cs_t1, cs_t2, knots_1, knots_2,
    exercise_start_time, P_sa_store, theta_ao_store, theta_po_store,
    theta_mi_store, theta_tr_store, V_lv_store, V_rv_store, P_rv_store,
    P_la_store, V_la_store, V_ra_store, P_ra_store, P_lv_store,
    phi_atr_store, phi_store, tidal_store, VAflow_store, Q_pp_store,
    dP_rv_dt_store, dP_lv_dt_store, P_peri_store,
):
    """
        Main model derivatives function with improved organization
        Coordinates all system computations and updates
        """

    (time_since_beat,
     HR, Vu_ev, Vu_sv, Vu_rmv, Vu_amv,
     Emax_lv, Emax_rv, f_sp, f_sh, f_v, f_sv, phi_met, HR_every, Vu_ev_every, Vu_sv_every,
     Vu_rmv_every, Vu_amv_every, Emax_lv_every, Emax_rv_every,
     prev_flat_bit,

     # for targets
     VT_lv, VT_rv, P_sa, P_rv, P_la, VT_la, VT_ra, P_ra, P_lv, phi, phi_atr, V, VAflow, Q_pp, theta_ao, theta_po, theta_mi, theta_tr, dP_rv_dt, dP_lv_dt, P_peri,

     Pa_O2, Pa_CO2, Pb_CO2,
     PA_O2, PA_CO2, Nt,
     Pa_O2_art_target_every, Pa_CO2_art_target_every,

     t1, t2, finish_breath_time, PamO2, PamCO2, PmbCO2,

     dVT_pa_dt, dVT_pp_dt, dVT_pv_dt, dQ_pa_dt, dVT_la_dt, dVT_lv_dt, dVT_ra_dt, dVT_rv_dt, dVT_sv_dt,
     dVT_bv_dt, dVT_hv_dt, dVT_rmv_dt, dVT_amv_dt, dP_sp_dt, dP_sa_dt, dQ_sa_dt, dVT_vc_dt,
     dtheta_ao_dt, d2theta_ao_dt2, dtheta_po_dt, d2theta_po_dt2, dtheta_mi_dt, d2theta_mi_dt2, dtheta_tr_dt,
     d2theta_tr_dt2,

     # cardio controller derivatives
     dtheta_change_O2_sp_dt, dtheta_change_CO2_sp_dt, dtheta_change_O2_sv_dt, dtheta_change_CO2_sv_dt,
     dtheta_change_O2_sh_dt, dtheta_change_CO2_sh_dt, dP_tilda_dt, d_fac_dt, df_ap_dt, dR_ep_change_dt,
     dR_sp_change_dt, dR_rmp_n_change_dt, dR_amp_n_change_dt, dVu_ev_change_dt, dVu_sv_change_dt,
     dVu_rmv_change_dt, dVu_amv_change_dt, dEmax_lv_change_dt, dEmax_rv_change_dt, d_Ts_change_dt,
     d_Tv_change_dt, dxb_O2_dt, dxb_CO2_dt, dxh_O2_dt, dxh_CO2_dt, dWh_dt, dxrm_O2_dt, dxrm_CO2_dt, dxam_O2_dt,
     dxM_dt, dx_met_dt, dP_n_current_dt,

     # gas exchange derivatives
     dPd_1_O2_dt, dPd_1_CO2_dt, dPd_2_O2_dt, dPd_2_CO2_dt, dPd_3_O2_dt, dPd_3_CO2_dt, dPd_4_O2_dt,
     dPd_4_CO2_dt, dPd_5_O2_dt, dPd_5_CO2_dt, dPa_O2_dt, dPa_CO2_dt, d2Pa_O2_dt2, d2Pa_CO2_dt2, dPA_O2_dt,
     dPA_CO2_dt, dPCSFCO2_dt, dMRTO2_dt, dMRTCO2_dt, dCTO2_dt, dCvtCO2_dt, dCBO2_dt, dCvbCO2_dt, dMRV_dt,

     # resp control derivatives
     d_VE_integral_dt


     ) = njit_compatible(t, state, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters, HR_store,
                      time_since_beat_store, HR_every_store, Vu_ev_every_store,
                      Vu_sv_every_store, Vu_rmv_every_store, Vu_amv_every_store, Emax_lv_every_store,
                      Emax_rv_every_store,
                      Vu_ev_store, Vu_sv_store, Vu_rmv_store, Vu_amv_store, Emax_lv_store, Emax_rv_store,
                      f_sp_history, f_sh_history, f_v_history, f_sv_history, phi_met_history,
                      Pa_O2_art_target_every_store, Pa_CO2_art_target_every_store, Nt_store, prev_flat_bit_store, finish_breath_time_store,
                      Pa_O2_every_store,
                      Pa_CO2_every_store, Pb_CO2_every_store, PamO2_store, PamCO2_store, PmbCO2_store, t1_store,
                      t2_store, cs_t1, cs_t2, knots_1, knots_2, exercise_start_time)

    store_index = (i - num_removed) % BUFFER_LIMIT

    # Cardiovascular Controller
    # update values needed in other systems
    time_since_beat_store[store_index] = time_since_beat
    HR_store[store_index] = HR
    Vu_ev_store[store_index] = Vu_ev
    Vu_sv_store[store_index] = Vu_sv
    Vu_rmv_store[store_index] = Vu_rmv
    Vu_amv_store[store_index] = Vu_amv
    Emax_lv_store[store_index] = Emax_lv
    Emax_rv_store[store_index] = Emax_rv
    f_sp_history[store_index] = f_sp
    f_sh_history[store_index] = f_sh
    f_v_history[store_index] = f_v
    f_sv_history[store_index] = f_sv
    phi_met_history[store_index] = phi_met
    HR_every_store[store_index] = HR_every
    Vu_ev_every_store[store_index] = Vu_ev_every
    Vu_sv_every_store[store_index] = Vu_sv_every
    Vu_rmv_every_store[store_index] = Vu_rmv_every
    Vu_amv_every_store[store_index] = Vu_amv_every
    Emax_lv_every_store[store_index] = Emax_lv_every
    Emax_rv_every_store[store_index] = Emax_rv_every
    P_sa_store[store_index] = P_sa
    theta_ao_store[store_index] = theta_ao
    theta_po_store[store_index] = theta_po
    theta_mi_store[store_index] = theta_mi
    theta_tr_store[store_index] = theta_tr
    V_lv_store[store_index] = VT_lv
    V_rv_store[store_index] = VT_rv
    P_rv_store[store_index] = P_rv
    P_la_store[store_index] = P_la
    V_la_store[store_index] = VT_la
    V_ra_store[store_index] = VT_ra
    P_ra_store[store_index] = P_ra
    P_lv_store[store_index] = P_lv
    phi_atr_store[store_index] = phi_atr
    phi_store[store_index] = phi
    tidal_store[store_index] = V
    VAflow_store[store_index] = VAflow
    Q_pp_store[store_index] = Q_pp
    dP_rv_dt_store[store_index] = dP_rv_dt
    dP_lv_dt_store[store_index] = dP_lv_dt
    P_peri_store[store_index] = P_peri
    prev_flat_bit_store[store_index] = prev_flat_bit
    t1_store[store_index] = t1
    t2_store[store_index] = t2

    # gas
    # update values needed in other systems
    Pa_O2_every_store[store_index] = Pa_O2
    Pa_CO2_every_store[store_index] = Pa_CO2
    Pb_CO2_every_store[store_index] = Pb_CO2
    Pa_O2_art_target_every_store[store_index] = Pa_O2_art_target_every
    Pa_CO2_art_target_every_store[store_index] = Pa_CO2_art_target_every
    Nt_store[store_index] = Nt

    # resp control
    finish_breath_time_store[store_index] = finish_breath_time
    PamO2_store[store_index] = PamO2
    PamCO2_store[store_index] = PamCO2
    PmbCO2_store[store_index] = PmbCO2


    return np.array((  # cardio derivatives
        dVT_pa_dt, dVT_pp_dt, dVT_pv_dt, dQ_pa_dt, dVT_la_dt, dVT_lv_dt, dVT_ra_dt, dVT_rv_dt, dVT_sv_dt,
        dVT_bv_dt, dVT_hv_dt, dVT_rmv_dt, dVT_amv_dt, dP_sp_dt, dP_sa_dt, dQ_sa_dt, dVT_vc_dt,
        dtheta_ao_dt, d2theta_ao_dt2, dtheta_po_dt, d2theta_po_dt2, dtheta_mi_dt, d2theta_mi_dt2, dtheta_tr_dt,
        d2theta_tr_dt2,

        # cardio controller derivatives
        dtheta_change_O2_sp_dt, dtheta_change_CO2_sp_dt, dtheta_change_O2_sv_dt, dtheta_change_CO2_sv_dt,
        dtheta_change_O2_sh_dt, dtheta_change_CO2_sh_dt, dP_tilda_dt, d_fac_dt, df_ap_dt, dR_ep_change_dt,
        dR_sp_change_dt, dR_rmp_n_change_dt, dR_amp_n_change_dt, dVu_ev_change_dt, dVu_sv_change_dt,
        dVu_rmv_change_dt, dVu_amv_change_dt, dEmax_lv_change_dt, dEmax_rv_change_dt, d_Ts_change_dt,
        d_Tv_change_dt, dxb_O2_dt, dxb_CO2_dt, dxh_O2_dt, dxh_CO2_dt, dWh_dt, dxrm_O2_dt, dxrm_CO2_dt, dxam_O2_dt,
        dxM_dt, dx_met_dt, dP_n_current_dt,

        # gas exchange derivatives
        dPd_1_O2_dt, dPd_1_CO2_dt, dPd_2_O2_dt, dPd_2_CO2_dt, dPd_3_O2_dt, dPd_3_CO2_dt, dPd_4_O2_dt,
        dPd_4_CO2_dt, dPd_5_O2_dt, dPd_5_CO2_dt, dPa_O2_dt, dPa_CO2_dt, d2Pa_O2_dt2, d2Pa_CO2_dt2, dPA_O2_dt,
        dPA_CO2_dt, dPCSFCO2_dt, dMRTO2_dt, dMRTCO2_dt, dCTO2_dt, dCvtCO2_dt, dCBO2_dt, dCvbCO2_dt, dMRV_dt,

        # resp control derivatives
        d_VE_integral_dt
    ))


_RHS_READ_KEYS = (
    "HR_store", "time_since_beat_store", "HR_every_store",
    "Vu_ev_every_store", "Vu_sv_every_store", "Vu_rmv_every_store",
    "Vu_amv_every_store", "Emax_lv_every_store", "Emax_rv_every_store",
    "Vu_ev_store", "Vu_sv_store", "Vu_rmv_store", "Vu_amv_store",
    "Emax_lv_store", "Emax_rv_store", "f_sp_store", "f_sh_store",
    "f_v_store", "f_sv_store", "phi_met_store",
    "Pa_O2_art_target_every_store", "Pa_CO2_art_target_every_store",
    "Nt_store", "prev_flat_bit_store", "finish_breath_time",
    "Pa_O2_every_store", "Pa_CO2_every_store", "Pb_CO2_every_store",
    "PamO2", "PamCO2", "PmbCO2", "t1_store", "t2_store",
)

_RHS_WRITE_KEYS = (
    "P_sa_store", "theta_ao_store", "theta_po_store", "theta_mi_store",
    "theta_tr_store", "V_lv_store", "V_rv_store", "P_rv_store",
    "P_la_store", "V_la_store", "V_ra_store", "P_ra_store",
    "P_lv_store", "phi_atr_store", "phi_store", "tidal_store",
    "VAflow_store", "Q_pp_store", "dP_rv_dt_store", "dP_lv_dt_store",
    "P_peri_store",
)


def make_wrapper(updates):
    """Bind a simulation's storage arrays into a lightweight SciPy RHS."""
    read_arrays = tuple(updates[key] for key in _RHS_READ_KEYS)
    write_arrays = tuple(updates[key] for key in _RHS_WRITE_KEYS)

    def fast_model_derivatives(
        t, state, unused_updates, num_removed, i, BUFFER_LIMIT, all_time,
        Input_Parameters, cs_t1, cs_t2, knots_1, knots_2,
        exercise_start_time,
    ):
        return model_derivatives_njit(
            t, state, num_removed, i, BUFFER_LIMIT, all_time,
            Input_Parameters, *read_arrays, cs_t1, cs_t2, knots_1, knots_2,
            exercise_start_time, *write_arrays,
        )

    return fast_model_derivatives


def model_derivatives(
    t, state, updates, num_removed, i, BUFFER_LIMIT, all_time,
    Input_Parameters, cs_t1, cs_t2, knots_1, knots_2, exercise_start_time,
):
    """Compatibility entry point for callers not yet using make_wrapper."""
    read_arrays = tuple(updates[key] for key in _RHS_READ_KEYS)
    write_arrays = tuple(updates[key] for key in _RHS_WRITE_KEYS)
    return model_derivatives_njit(
        t, state, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters,
        *read_arrays, cs_t1, cs_t2, knots_1, knots_2, exercise_start_time,
        *write_arrays,
    )
