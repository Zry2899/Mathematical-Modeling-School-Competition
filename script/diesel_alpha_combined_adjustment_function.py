"""
Combined diesel alpha function.

This function stitches together the upward- and downward-adjustment empirical
forms. It is intended to approximate the six-loss equal-weight welfare-optimal
diesel alpha.
"""

from __future__ import annotations

import math


DIESEL_COMBINED_UPWARD_ALPHA_PARAMS = {
    "intercept": 0.893319718496,
    "beta_fd_log": -0.0406656108272,
    "beta_fg_hinge": 0.0282566081518,
    "beta_s_d": 0.0182799769857,
    "beta_prev_u_d": 0.00787431396476,
    "beta_cpi_low": -0.020523704782,
    "beta_cpi_high": 0.031976841653,
    "fd_threshold": 40,
    "fg_threshold": 0,
    "cpi_low": 0.02,
    "cpi_high": 0.03,
    "fd_log_center": 4.80599736939,
    "fg_hinge_center": 1.67832628657,
    "S_d_center": 18.3988140967,
    "prev_u_d_center": 0,
    "cpi_low_gap_center": 0.2,
    "cpi_high_gap_center": 0,
}

DIESEL_COMBINED_DOWNWARD_ALPHA_PARAMS = {
    "intercept": 1.19739540279,
    "beta_abs_fd_log": 0.2541631287,
    "beta_s_d": -0.0668703378625,
    "beta_prev_u_d": -0.0161246829237,
    "fd_abs_threshold": 0,
    "abs_fd_log_center": 5.25069001022,
    "S_d_center": 18.3988140967,
    "prev_u_d_center": 0,
}


def clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def diesel_alpha_upward_component(
    f_gasoline: float,
    f_diesel: float,
    S_diesel_lag: float,
    prev_u_diesel: float,
    cpi_yoy: float,
    params: dict[str, float] | None = None,
) -> float:
    p = DIESEL_COMBINED_UPWARD_ALPHA_PARAMS if params is None else params
    fd_log = math.log1p(max(float(f_diesel) - p["fd_threshold"], 0.0)) - p["fd_log_center"]
    fg_hinge = max(float(f_gasoline) - p["fg_threshold"], 0.0) / 100.0 - p["fg_hinge_center"]
    s_d = (float(S_diesel_lag) - p["S_d_center"]) / 10.0
    prev_u_d = (float(prev_u_diesel) - p["prev_u_d_center"]) / 100.0
    cpi_low_gap = max(p["cpi_low"] - float(cpi_yoy), 0.0) / 0.01 - p["cpi_low_gap_center"]
    cpi_high_gap = max(float(cpi_yoy) - p["cpi_high"], 0.0) / 0.01 - p["cpi_high_gap_center"]
    alpha = (
        p["intercept"]
        + p["beta_fd_log"] * fd_log
        + p["beta_fg_hinge"] * fg_hinge
        + p["beta_s_d"] * s_d
        + p["beta_prev_u_d"] * prev_u_d
        + p["beta_cpi_low"] * cpi_low_gap
        + p["beta_cpi_high"] * cpi_high_gap
    )
    return clip01(alpha)


def diesel_alpha_downward_component(
    f_diesel: float,
    S_diesel_lag: float,
    prev_u_diesel: float,
    params: dict[str, float] | None = None,
) -> float:
    p = DIESEL_COMBINED_DOWNWARD_ALPHA_PARAMS if params is None else params
    abs_fd_log = (
        math.log1p(max(abs(float(f_diesel)) - p["fd_abs_threshold"], 0.0))
        - p["abs_fd_log_center"]
    )
    s_d = (float(S_diesel_lag) - p["S_d_center"]) / 10.0
    prev_u_d = (float(prev_u_diesel) - p["prev_u_d_center"]) / 100.0
    alpha = (
        p["intercept"]
        + p["beta_abs_fd_log"] * abs_fd_log
        + p["beta_s_d"] * s_d
        + p["beta_prev_u_d"] * prev_u_d
    )
    return clip01(alpha)


def diesel_alpha_adjustment(
    f_gasoline: float,
    f_diesel: float,
    S_diesel_lag: float,
    prev_u_diesel: float,
    cpi_yoy: float,
    upward_params: dict[str, float] | None = None,
    downward_params: dict[str, float] | None = None,
) -> float:
    """
    Calculate diesel alpha for either upward or downward theoretical adjustment.

    Returns NaN when f_diesel is zero because alpha has no economic meaning
    when u = alpha * f and f = 0.
    """
    f_d = float(f_diesel)
    if f_d > 0:
        return diesel_alpha_upward_component(
            f_gasoline=f_gasoline,
            f_diesel=f_d,
            S_diesel_lag=S_diesel_lag,
            prev_u_diesel=prev_u_diesel,
            cpi_yoy=cpi_yoy,
            params=upward_params,
        )
    if f_d < 0:
        return diesel_alpha_downward_component(
            f_diesel=f_d,
            S_diesel_lag=S_diesel_lag,
            prev_u_diesel=prev_u_diesel,
            params=downward_params,
        )
    return float("nan")
