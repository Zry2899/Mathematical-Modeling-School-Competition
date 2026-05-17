"""
Gasoline downward-regulation alpha function.

This module provides the fitted empirical function used to approximate the
six-loss equal-weight welfare-optimal gasoline alpha under a downward
theoretical adjustment scenario.
"""

from __future__ import annotations

import math


GASOLINE_DOWNWARD_ADJUSTMENT_ALPHA_PARAMS = {
    "intercept": 0.981202111683,
    "beta_abs_fg_log": 0.0673511088237,
    "beta_s_g": -0.0165639223198,
    "beta_prev_u_g": -0.0148596217634,
    "fg_abs_threshold": 0,
    "abs_fg_log_center": 5.28640753694,
    "S_g_center": 15.2561563831,
    "prev_u_g_center": 0,
}


def clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def gasoline_alpha_downward_adjustment(
    f_gasoline: float,
    S_gasoline_lag: float,
    prev_u_gasoline: float,
    params: dict[str, float] | None = None,
) -> float:
    """
    Calculate gasoline alpha under a downward theoretical-adjustment scenario.

    Parameters
    ----------
    f_gasoline:
        Gasoline theoretical adjustment from the first model. This downward
        function is intended for f_gasoline < 0.
    S_gasoline_lag:
        Lagged accumulated gasoline transmission pressure.
    prev_u_gasoline:
        Previous gasoline price adjustment.
    params:
        Optional parameter dictionary. Defaults to the fitted parameters above.
    """
    p = GASOLINE_DOWNWARD_ADJUSTMENT_ALPHA_PARAMS if params is None else params

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
