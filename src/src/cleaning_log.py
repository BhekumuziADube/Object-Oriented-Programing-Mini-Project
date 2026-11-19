"""CleaningLog: a numbered record of every cleaning action.

Card CL1. Author: Shepard.

Other cards write into the log:
    C6  (Preprocessor)  logs each preprocessing action
    CL2 (clean datasets) logs each export and verification step
The finished log is saved as docs/cleaning_log.csv and summarised in the report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd


@dataclass(frozen=True)
class LogEntry:
    """One cleaning action. Frozen, so an entry cannot be edited afterwards."""

    step: int
    dimension: str
    column: str
    issue: str
    action: str
    rows_affected: int
    justification: str


class CleaningLog:
    """Collect log entries in order and save them as a CSV file."""

    DIMENSIONS: Tuple[str, ...] = (
        "Completeness",
        "Timeliness",
        "Consistency",
        "Validity",
        "Uniqueness",
        "Accuracy",
    )
    COLUMNS: List[str] = [
        "step",
        "dimension",
        "column",
        "issue",
        "action",
        "rows_affected",
        "justification",
    ]

    def __init__(self) -> None:
        self._entries: List[LogEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> Tuple[LogEntry, ...]:
        """All entries as a read-only tuple."""
        return tuple(self._entries)

    def to_frame(self) -> pd.DataFrame:
        """The log as a table (has the right columns even when empty)."""
        return pd.DataFrame(
            [asdict(e) for e in self._entries], columns=self.COLUMNS
        )

    def save(self, path: Union[str, Path] = "docs/cleaning_log.csv") -> Path:
        """Write the log to a CSV file, creating the folder if needed."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(target, index=False)
        return target
