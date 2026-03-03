"""write.py

Small CSV I/O helper for writing tabular artifacts.
"""

from pathlib import Path
from typing import Any, Dict
import pandas as pd


def append_csv_row(csv_path: Path | str, row: Dict[str, Any]) -> None:
    """
    Appends a single row to a CSV runs file, creating the file and header if needed.

    Parameters
    ----------
        csv_path : Path | str
            Path to the CSV runs file.
        row : Dict[str, Any]
            Row data to append as a single record.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row])
    df_new.to_csv(csv_path,
                  mode="a",
                  header=(not csv_path.exists()),
                  index=False)
