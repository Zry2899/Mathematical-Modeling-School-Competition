"""
Gasoline upward-regulation alpha function.

This module provides the fitted empirical function used to approximate the
six-loss equal-weight welfare-optimal gasoline alpha under an upward theoretical
adjustment scenario.
"""

from __future__ import annotations

import math


GASOLINE_UPWARD_ADJUSTMENT_ALPHA_PARAMS = {
    "intercept": 0.931258911977,
    "beta_fg_log": -0.0307497531556,
    "beta_fd_hinge": 0.0243146017732,
    "beta_s_g": 0.015487118973,
    "beta_prev_u_g": 0.0132194195919,
    "beta_cpi_low": 0.094439939355,
    "beta_cpi_high": -0.0789743433432,
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


def clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def gasoline_alpha_upward_adjustment(
    f_gasoline: float,
    f_diesel: float,
    S_gasoline_lag: float,
    prev_u_gasoline: float,
    cpi_yoy: float,
    params: dict[str, float] | None = None,
) -> float:
    """
    Calculate gasoline alpha under an upward theoretical-adjustment scenario.

    Parameters
    ----------
    f_gasoline:
        Gasoline theoretical adjustment from the first model. This upward
        function is intended for f_gasoline > 0.
    f_diesel:
        Diesel theoretical adjustment. In the upward scenario it is usually
        positive and enters through a threshold-linear cross term.
    S_gasoline_lag:
        Lagged accumulated gasoline transmission pressure.
    prev_u_gasoline:
        Previous gasoline price adjustment.
    cpi_yoy:
        CPI year-on-year rate, expressed as a decimal.
    params:
        Optional parameter dictionary. Defaults to the fitted parameters above.
    """
    p = GASOLINE_UPWARD_ADJUSTMENT_ALPHA_PARAMS if params is None else params

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


# Backward-compatible aliases. Prefer the explicit upward-adjustment names above
# when adding the later downward-adjustment function.
GASOLINE_UPWARD_ALPHA_PARAMS = GASOLINE_UPWARD_ADJUSTMENT_ALPHA_PARAMS
gasoline_alpha_upward = gasoline_alpha_upward_adjustment
