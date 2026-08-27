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
    a2 = (-P_ao - E_rs * VA * (t1 + t2) - E_rs * VD) / (t1 ** 2)
    a1 = -2 * a2 * t1
    Pt1 = a1 * t1 + a2 * (t1 ** 2)
    Vt1 = VA * (t1 + t2) + VD
    # tau = max((t2 / (-np.log(tolerance * R_rs / Pt1))), 0.001)
    raw_tau = 0.5 * t2 / (-np.log(tolerance * R_rs / Pt1))
    tau_min = 0.001
    tau = np.sqrt(raw_tau * raw_tau + tau_min * tau_min)
    B = E_rs / R_rs

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

    P_musc_exp = Pt1 * np.exp(-((z - t1) ** 2) / tau)
    dP_musc_dt_exp = P_musc_exp * (-2 * (z - t1) / tau)


    # Compute constants for Volume solution
    c1 = (Vt1 - ((a1 / E_rs) * t1 + (a2 / E_rs) * (t1 ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t1)) / (
            np.exp(-B * t1) - 1)

    d1 = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) - c1

    # Calculate for 0 <= times <= t1
    V_insp = ((a1 / E_rs) * x - (a1 * R_rs / (E_rs ** 2)) +
                     (a2 / E_rs) * (x ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * x +
                     (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) +
                     c1 * np.exp(-B * x) + d1)

    # Compute constants
    mu = t1 + 0.5 * B * tau
    term1 = - (t1 * t1) / tau
    term2 = (mu ** 2) / tau
    K = term1 + term2
    pref = np.exp(K) * 0.5 * np.sqrt(np.pi * tau)

    I0 = gaussian_integral(mu, t1, pref, tau)

    I_z = np.zeros(len(z))
    for i in range(len(z)):
        I_z[i] = gaussian_integral(mu, z[i], pref, tau)

    integral = I_z - I0
    constant = (Vt1 / np.exp(-B * t1))  # - (Pt1 / R_rs) * I0
    expBz = np.exp(-B * z)
    V_exp = (Pt1 / R_rs) * expBz * integral + constant * expBz


    dV_dt_insp = (P_musc_insp - E_rs * V_insp) / R_rs
    dV_dt_exp = (P_musc_exp - E_rs * V_exp) / R_rs


    # continue calculations
    dV2_dt2_values_squared_insp = ((1 / R_rs) * (dP_musc_dt_insp - E_rs * dV_dt_insp)) ** 2
    dV2_dt2_values_squared_exp = ((1 / R_rs) * (dP_musc_dt_exp - E_rs * dV_dt_exp)) ** 2


    E1_n_insp = (1 - np.clip((P_musc_insp / Pmax), 0, 0.999999)) ** n
    E2_n_insp = (1 - np.clip((np.abs(dP_musc_dt_insp) / Pmax_dot), 0, 0.999999)) ** n


    integrand_inspire = (P_musc_insp * dV_dt_insp) / (
    E1_n_insp * E2_n_insp) + lambda1 * dV2_dt2_values_squared_insp

    integrand_expire = dV2_dt2_values_squared_exp

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
        c1 = (Vt1 - ((a1 / E_rs) * t1 + (a2 / E_rs) * (t1 ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t1)) / (
                np.exp(-B * t1) - 1)
        d1 = (a1 * R_rs / (E_rs ** 2)) - (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) - c1

        V = ((a1 / E_rs) * t - (a1 * R_rs / (E_rs ** 2)) +
             (a2 / E_rs) * (t ** 2) - (2 * a2 * R_rs / (E_rs ** 2)) * t +
             (2 * a2 * (R_rs ** 2) / (E_rs ** 3)) +
             c1 * expBz + d1)
        dV_dt = (1 / R_rs) * (a1 * t + a2 * (t ** 2) - E_rs * V)
    else:
        mu = t1 + 0.5 * B * tau
        term1 = - (t1 * t1) / tau
        term2 = (mu ** 2) / tau
        K = term1 + term2
        pref = np.exp(K) * 0.5 * np.sqrt(np.pi * tau)

        I0 = gaussian_integral(mu, t1, pref, tau)
        I_z = gaussian_integral(mu, t, pref, tau)

        integral = I_z - I0
        constant = (Vt1 / np.exp(-B * t1))
        V = (Pt1 / R_rs) * expBz * integral + constant * expBz

        P_musc = Pt1 * np.exp((-(t - t1) ** 2) / tau)
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
