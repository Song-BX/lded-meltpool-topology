from __future__ import annotations

import pandas as pd

from .config import CANONICAL_AGGREGATION


def build_temporal_context(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep the gradient time series separate from the pre-specified Comment 1 audit."""
    context = metrics.loc[metrics["aggregation_strategy"] == CANONICAL_AGGREGATION].copy()
    context["context_role"] = "post_hoc_comment9_context"
    context["comment1_temporal_criterion"] = False
    return context.sort_values(["time_s", "power_W", "region"]).reset_index(drop=True)

