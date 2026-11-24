import math
import numpy as np
from matplotlib import pyplot as plt
from numba import njit

# outside objective, decide once:
t1_upper_bound = 5
t2_upper_bound = 7
T_max = t1_upper_bound + t2_upper_bound
n_steps = int(np.round(T_max / 0.001)) + 1
base_times = np.linspace(0, T_max, n_steps)


@njit
def compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance):
    """
    Compute constants for the respiratory model
    """
    # Precompute key values
    B = E_rs / R_rs
    Vt1 = VA * (t1 + t2) + VD
    F1 = t1 / E_rs - R_rs / (E_rs ** 2) + R_rs / (E_rs ** 2) * np.exp(-B * t1)
    F2 = t1 ** 2 / E_rs - 2 * R_rs * t1 / (E_rs ** 2) + 2 * R_rs ** 2 / (E_rs ** 3) - 2 * R_rs ** 2 / (E_rs ** 3) * np.exp(-B * t1)
    a1 = (Vt1 * (1 - E_rs * F2 / (t1 ** 2))) / (F1 - F2 / t1)
    a2 = (E_rs * Vt1 / (t1 ** 2)) - a1 / t1
    Pt1 = a1 * t1 + a2 * (t1 ** 2)
    # tau = max((t2 / (-np.log(tolerance * R_rs / Pt1))), 0.001)
    raw_tau = (t2 * t2) / (-np.log(tolerance * R_rs / Pt1))
    tau_min = 0.001
    tau = np.sqrt(raw_tau * raw_tau + tau_min * tau_min)

    return a1, a2, Pt1, Vt1, tau, B

@njit
def trapz_uniform(y, dt):
    s = 0.0
    for i in range(1, len(y)):
        s += 0.5 * (y[i] + y[i-1])
    return s * dt


@njit
def gaussian_integral(mu, z, pref, tau):
    # erf argument
    arg = (z - mu) / np.sqrt(tau)

    # full integral
    return pref * math.erf(arg)


# @njit
# def gaussian_expiration_V(t, t1, Vt1, Pt1, tau, B, R_rs):
#     """
#     Expiratory volume driven by the Gaussian (diaphragm braking) pressure only
#     """
#     mu = t1 + 0.5 * B * tau
#     K = - (t1 * t1) / tau + (mu ** 2) / tau
#     pref = np.exp(K) * 0.5 * np.sqrt(np.pi * tau)
#     integral = gaussian_integral(mu, t, pref, tau) - gaussian_integral(mu, t1, pref, tau)
#     constant = (Vt1 / np.exp(-B * t1))
#     expBz = np.exp(-B * t)
#     return (Pt1 / R_rs) * expBz * integral + constant * expBz

@njit
def erfcx_pos(x):  # exp(x^2) * erfc(x) for x >= 0
    if x < 25.0:
        return math.erfc(x) * math.exp(x * x)
    x2 = x * x
    return (1.0 - 0.5 / x2 + 0.75 / (x2 * x2) - 1.875 / (x2 * x2 * x2)) / (x * math.sqrt(math.pi))

@njit
def gaussian_expiration_V(t, t1, Vt1, Pt1, tau, B, R_rs):
    """
    Expiratory volume driven by the Gaussian (diaphragm braking) pressure only.
    Written with erfcx: the erf form above cancels to noise when B * sqrt(tau) is large.
    """
    s = t - t1
    sq = math.sqrt(tau)
    c = 0.5 * B * sq
    x = c - s / sq
    if x >= 0.0:
        term = math.exp(-s * s / tau) * erfcx_pos(x)
    else:
        term = 2.0 * math.exp(c * c - B * s) - math.exp(-s * s / tau) * erfcx_pos(-x)
    eBs = math.exp(-B * s)
    return (Pt1 / R_rs) * 0.5 * math.sqrt(math.pi * tau) * (term - eBs * erfcx_pos(c)) + Vt1 * eBs


@njit
def poly_response(s, E_rs, R_rs):
    """
    Volume responses (from zero) to muscle pressures s^2 and s^3
    """
    B = E_rs / R_rs
    decay = 1 - np.exp(-B * s)
    F2 = s ** 2 / E_rs - 2 * R_rs * s / (E_rs ** 2) + 2 * R_rs ** 2 / (E_rs ** 3) * decay
    F3 = s ** 3 / E_rs - 3 * R_rs * s ** 2 / (E_rs ** 2) + 6 * R_rs ** 2 * s / (E_rs ** 3) - 6 * R_rs ** 3 / (E_rs ** 4) * decay
    return F2, F3


@njit
def expiratory_drive(t1, t2, Vt1, Pt1, tau, B, E_rs, R_rs, V_end):
    """
    Abdominal (expiratory) muscle pressure b2 * s^2 + b3 * s^3, s = t - t1.
    b2, b3 give V(t1 + t2) = V_end and zero flow at t1 + t2, as a1, a2 do for inspiration.
    """
    F2, F3 = poly_response(t2, E_rs, R_rs)
    Dv = V_end - gaussian_expiration_V(t1 + t2, t1, Vt1, Pt1, tau, B, R_rs)
    Dp = E_rs * V_end - Pt1 * np.exp(-(t2 ** 2) / tau)
    b2 = (Dv - Dp * F3 / (t2 ** 3)) / (F2 - F3 / t2)
    b3 = Dp / (t2 ** 3) - b2 / t2
    return b2, b3


@njit
def calculate_variables(initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao, Pmax, Pmax_dot, n, lambda1):
    """
    Updated method for calculating P_musc and dP_musc/dt
    """
    t1, t2 = initial_guess
    a1, a2, Pt1, Vt1, tau, B = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)

    N1 = 1000
    N2 = 1000
    #
    # x = np.linspace(0.0, t1, N1)
    # z = np.linspace(t1, t1 + t2, N2)
    #
    # dt1 = x[1] - x[0]
    # dt2 = z[1] - z[0]

    s1 = np.linspace(0.0, 1.0, N1)
    s2 = np.linspace(0.0, 1.0, N2)

    x = t1 * s1
    z = t1 + t2 * s2
    dt1 = t1 / (N1 - 1)
    dt2 = t2 / (N2 - 1)

    P_musc_insp = a1 * x + a2 * x ** 2
    dP_musc_dt_insp = a1 + 2 * a2 * x

    s_exp = z - t1
    b2, b3 = expiratory_drive(t1, t2, Vt1, Pt1, tau, B, E_rs, R_rs, 0.0)
    P_brake = Pt1 * np.exp(-(s_exp ** 2) / tau)
    P_musc_exp = P_brake + b2 * s_exp ** 2 + b3 * s_exp ** 3
    # dP_musc_dt_exp = P_brake * (-2 * s_exp / tau) + 2 * b2 * s_exp + 3 * b3 * s_exp ** 2


    # Compute constants for Volume solution
    c = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3))

    # Calculate for 0 <= times <= t1
    V_insp = ((a2 / E_rs) * (x ** 2) + (a1 / E_rs - (2 * a2 * R_rs / (E_rs ** 2))) * x -
              (a1 * R_rs / E_rs ** 2) + 2 * a2 * (R_rs ** 2) / (E_rs ** 3) +
                     c * np.exp(-B * x))

    # Calculate for t1 <= times <= t1 + t2: Gaussian brake + abdominal drive
    V_brake = np.empty(len(z))
    for i in range(len(z)):
        V_brake[i] = gaussian_expiration_V(z[i], t1, Vt1, Pt1, tau, B, R_rs)
    F2_s, F3_s = poly_response(s_exp, E_rs, R_rs)
    V_exp = V_brake + b2 * F2_s + b3 * F3_s


    dV_dt_insp = (P_musc_insp - E_rs * V_insp) / R_rs
    dV_dt_exp = (P_musc_exp - E_rs * V_exp) / R_rs


    # continue calculations
    dV2_dt2_values_squared_insp = ((1 / R_rs) * (dP_musc_dt_insp - E_rs * dV_dt_insp)) ** 2
    # dV2_dt2_values_squared_exp = ((1 / R_rs) * (dP_musc_dt_exp - E_rs * dV_dt_exp)) ** 2  # old expiratory smoothness cost


    E1_n_insp = (1 - np.clip((P_musc_insp / Pmax), 0, 0.999999)) ** n
    E2_n_insp = (1 - np.clip((np.abs(dP_musc_dt_insp) / Pmax_dot), 0, 0.999999)) ** n


    integrand_inspire = (P_musc_insp * dV_dt_insp) / (
    E1_n_insp * E2_n_insp) + lambda1 * dV2_dt2_values_squared_insp

    # Expiratory cost: resistive work of expiration (energy dissipated pushing air out through R_rs)
    integrand_expire = R_rs * dV_dt_exp ** 2

    integral_inspire = trapz_uniform(integrand_inspire, dt1)
    integral_expire = trapz_uniform(integrand_expire, dt2)

    WI = (1 / (t1 + t2)) * integral_inspire
    WE = (1 / (t1 + t2)) * integral_expire

    return WI, WE


# @njit
def objective(initial_guess, required_params, VAflow, VD, dt, tolerance):
    """
    Optimized Objective function
    """
    lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao = required_params

    WI, WE = calculate_variables(initial_guess, VAflow, VD, tolerance, E_rs, R_rs, P_ao, Pmax, Pmax_dot, n, lambda1)

    t2_min = 0.5
    lambda_barrier = 1000.0
    barrier = lambda_barrier * np.log1p(np.exp(20.0 * (t2_min - initial_guess[1]))) / 20.0

    # Return cost function value
    return WI + lambda2 * WE + barrier


@njit
def calculate_single_V_dV_dt(t, initial_guess, VA, VD, tolerance, E_rs, R_rs, P_ao):
    """
    Updated method for calculating V and dV/dt values
    """
    # Precompute constants
    t1, t2 = initial_guess
    a1, a2, Pt1, Vt1, tau, B = compute_constants(t1, t2, VA, VD, E_rs, R_rs, P_ao, tolerance)
    expBz = np.exp(-B * t)

    if t <= t1:
        c = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3))

        # Calculate for 0 <= times <= t1
        V = ((a2 / E_rs) * (t ** 2) + (a1 / E_rs - (2 * a2 * R_rs / (E_rs ** 2))) * t -
                  (a1 * R_rs / E_rs ** 2) + 2 * a2 * (R_rs ** 2) / (E_rs ** 3) +
                  c * np.exp(-B * t))

        dV_dt = (1 / R_rs) * (a1 * t + a2 * (t ** 2) - E_rs * V)
    else:
        s_exp = t - t1
        b2, b3 = expiratory_drive(t1, t2, Vt1, Pt1, tau, B, E_rs, R_rs, 0.0)
        F2_s, F3_s = poly_response(s_exp, E_rs, R_rs)
        V = gaussian_expiration_V(t, t1, Vt1, Pt1, tau, B, R_rs) + b2 * F2_s + b3 * F3_s

        P_musc = Pt1 * np.exp(-(s_exp ** 2) / tau) + b2 * s_exp ** 2 + b3 * s_exp ** 3
        dV_dt = (P_musc - E_rs * V) / R_rs

    return V, dV_dt




 # # Calculate for t1 <= times <= t1 + t2
    # integral = integrate_z1_to_z1+z2(np.exp((-z ** 2 / tau) + (B + 2 * t1/tau) * z - (t1 ** 2) / tau))
    # constant = Vt1 * np.exp(B * z) - Pt1 / R_rs * integral
    # V[mask_t1_t2] = (Pt1/R_rs) * np.exp(-B * z) * integral + constant * np.exp(-B * z)

    # EXPIRATION USING GAUSSIAN INTEGRAL
    # Precompute z0 = 0 point for definite integral limits
    # a =  1 / tau
    # b = (B + 2 * t1 / tau)
    # c = (t1 ** 2) / tau
    #
    # V[mask_t1_t2] = np.sqrt(np.pi/a) * np.exp((b ** 2) / (4 * a) - c)
