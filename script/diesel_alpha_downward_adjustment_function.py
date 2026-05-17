"""
Diesel downward-regulation alpha function.

This module provides the fitted empirical function used to approximate the
six-loss equal-weight welfare-optimal diesel alpha under a downward theoretical
adjustment scenario.
"""

from __future__ import annotations

import math


DIESEL_DOWNWARD_ADJUSTMENT_ALPHA_PARAMS = {
    "intercept": 0.985876089533,
    "beta_abs_fd_log": 0.0881090798731,
    "beta_s_d": -0.0185058331911,
    "beta_prev_u_d": -0.0146246009487,
    "fd_abs_threshold": 0,
    "abs_fd_log_center": 5.25069001022,
    "S_d_center": 18.3988140967,
    "prev_u_d_center": 0,
}


def clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def diesel_alpha_downward_adjustment(
    f_diesel: float,
    S_diesel_lag: float,
    prev_u_diesel: float,
    params: dict[str, float] | None = None,
) -> float:
    """
    Calculate diesel alpha under a downward theoretical-adjustment scenario.

    Parameters
    ----------
    f_diesel:
        Diesel theoretical adjustment from the first model. This downward
        function is intended for f_diesel < 0.
    S_diesel_lag:
        Lagged accumulated diesel transmission pressure.
    prev_u_diesel:
        Previous diesel price adjustment.
    params:
        Optional parameter dictionary. Defaults to the fitted parameters above.
    """
    p = DIESEL_DOWNWARD_ADJUSTMENT_ALPHA_PARAMS if params is None else params

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
