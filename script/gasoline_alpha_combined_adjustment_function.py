"""
Combined gasoline alpha function.

This function stitches together the upward- and downward-adjustment empirical
forms. It is intended to approximate the six-loss equal-weight welfare-optimal
gasoline alpha.
"""

from __future__ import annotations

import math


GASOLINE_COMBINED_UPWARD_ALPHA_PARAMS = {
    "intercept": 0.885479082569,
    "beta_fg_log": -0.0311661869812,
    "beta_fd_hinge": 0.0462474878581,
    "beta_s_g": 0.0207029754575,
    "beta_prev_u_g": 0.0101731106195,
    "beta_cpi_low": 0.00742246172882,
    "beta_cpi_high": -0.0656565124115,
    "fg_threshold": 100,
    "fd_threshold": 90,
    "cpi_low": 0.02,
    "cpi_high": 0.03,
    "fg_log_center": 4.23167788625,
    "fd_hinge_center": 0.712413500226,
    "S_g_center": 15.2561563831,
    "prev_u_g_center": 0,
    "cpi_low_gap_center": 0.2,
    "cpi_high_gap_center": 0,
}

GASOLINE_COMBINED_DOWNWARD_ALPHA_PARAMS = {
    "intercept": 1.14032691457,
    "beta_abs_fg_log": 0.198835349505,
    "beta_s_g": -0.00688310912137,
    "beta_prev_u_g": -0.093174070378,
    "fg_abs_threshold": 0,
    "abs_fg_log_center": 5.28640753694,
    "S_g_center": 15.2561563831,
    "prev_u_g_center": 0,
}


def clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def gasoline_alpha_upward_component(
    f_gasoline: float,
    f_diesel: float,
    S_gasoline_lag: float,
    prev_u_gasoline: float,
    cpi_yoy: float,
    params: dict[str, float] | None = None,
) -> float:
    p = GASOLINE_COMBINED_UPWARD_ALPHA_PARAMS if params is None else params
    fg_log = math.log1p(max(float(f_gasoline) - p["fg_threshold"], 0.0)) - p["fg_log_center"]
    fd_hinge = max(float(f_diesel) - p["fd_threshold"], 0.0) / 100.0 - p["fd_hinge_center"]
    s_g = (float(S_gasoline_lag) - p["S_g_center"]) / 10.0
    prev_u_g = (float(prev_u_gasoline) - p["prev_u_g_center"]) / 100.0
    cpi_low_gap = max(p["cpi_low"] - float(cpi_yoy), 0.0) / 0.01 - p["cpi_low_gap_center"]
    cpi_high_gap = max(float(cpi_yoy) - p["cpi_high"], 0.0) / 0.01 - p["cpi_high_gap_center"]
    alpha = (
        p["intercept"]
        + p["beta_fg_log"] * fg_log
        + p["beta_fd_hinge"] * fd_hinge
        + p["beta_s_g"] * s_g
        + p["beta_prev_u_g"] * prev_u_g
        + p["beta_cpi_low"] * cpi_low_gap
        + p["beta_cpi_high"] * cpi_high_gap
    )
    return clip01(alpha)


def gasoline_alpha_downward_component(
    f_gasoline: float,
    S_gasoline_lag: float,
    prev_u_gasoline: float,
    params: dict[str, float] | None = None,
) -> float:
    p = GASOLINE_COMBINED_DOWNWARD_ALPHA_PARAMS if params is None else params
    abs_fg_log = (
        math.log1p(max(abs(float(f_gasoline)) - p["fg_abs_threshold"], 0.0))
        - p["abs_fg_log_center"]
    )
    s_g = (float(S_gasoline_lag) - p["S_g_center"]) / 10.0
    prev_u_g = (float(prev_u_gasoline) - p["prev_u_g_center"]) / 100.0
    alpha = (
        p["intercept"]
        + p["beta_abs_fg_log"] * abs_fg_log
        + p["beta_s_g"] * s_g
        + p["beta_prev_u_g"] * prev_u_g
    )
    return clip01(alpha)


def gasoline_alpha_adjustment(
    f_gasoline: float,
    f_diesel: float,
    S_gasoline_lag: float,
    prev_u_gasoline: float,
    cpi_yoy: float,
    upward_params: dict[str, float] | None = None,
    downward_params: dict[str, float] | None = None,
) -> float:
    """
    Calculate gasoline alpha for either upward or downward theoretical adjustment.

    Returns NaN when f_gasoline is zero because alpha has no economic meaning
    when u = alpha * f and f = 0.
    """
    f_g = float(f_gasoline)
    if f_g > 0:
        return gasoline_alpha_upward_component(
            f_gasoline=f_g,
            f_diesel=f_diesel,
            S_gasoline_lag=S_gasoline_lag,
            prev_u_gasoline=prev_u_gasoline,
            cpi_yoy=cpi_yoy,
            params=upward_params,
        )
    if f_g < 0:
        return gasoline_alpha_downward_component(
            f_gasoline=f_g,
            S_gasoline_lag=S_gasoline_lag,
            prev_u_gasoline=prev_u_gasoline,
            params=downward_params,
        )
    return float("nan")
