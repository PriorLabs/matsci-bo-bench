#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""The CSV reader every loader uses.

pandas' default float parser can land one bit away from the nearest double,
and where it does depends on the numpy build. That changes a pool's
fingerprint between machines. ``float_precision="round_trip"`` parses each
decimal to its nearest double, so a pool depends only on the file's bytes.
"""

from __future__ import annotations

import pandas as pd


def read_csv(*args, **kwargs) -> pd.DataFrame:
    """``pandas.read_csv`` with exact float parsing."""
    kwargs.setdefault("float_precision", "round_trip")
    return pd.read_csv(*args, **kwargs)
