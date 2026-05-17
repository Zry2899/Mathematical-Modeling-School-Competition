"""
Diesel upward-regulation alpha function.

This module provides the fitted empirical function used to approximate the
six-loss equal-weight welfare-optimal diesel alpha under an upward theoretical
adjustment scenario.
"""

from __future__ import annotations

import math


DIESEL_UPWARD_ADJUSTMENT_ALPHA_PARAMS = {
    "intercept": 0.899831036297,
    "beta_fd_log": -0.0328269125899,
    "beta_fg_hinge": 0.0120171320182,
    "beta_s_d": 0.0177287302323,
    "beta_prev_u_d": 0.0134960528484,
    "beta_cpi_low": -0.0510141614912,
    "beta_cpi_high": 0.0473632613059,
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


def clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def diesel_alpha_upward_adjustment(
    f_gasoline: float,
    f_diesel: float,
    S_diesel_lag: float,
    prev_u_diesel: float,
    cpi_yoy: float,
    params: dict[str, float] | None = None,
) -> float:
    """
    Calculate diesel alpha under an upward theoretical-adjustment scenario.

    Parameters
    ----------
    f_gasoline:
        Gasoline theoretical adjustment. In the upward scenario it is usually
        positive and enters through a threshold-linear cross term.
    f_diesel:
        Diesel theoretical adjustment from the first model. This upward
        function is intended for f_diesel > 0.
    S_diesel_lag:
        Lagged accumulated diesel transmission pressure.
    prev_u_diesel:
        Previous diesel price adjustment.
    cpi_yoy:
        CPI year-on-year rate, expressed as a decimal.
    params:
        Optional parameter dictionary. Defaults to the fitted parameters above.
    """
    p = DIESEL_UPWARD_ADJUSTMENT_ALPHA_PARAMS if params is None else params

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
