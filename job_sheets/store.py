"""In-memory batch store.

A batch is the set of :class:`JobSheet` objects produced from one upload,
addressed by a short ``batch_id`` used in the review/generate URLs. Kept in
memory for the MVP (no DB table); swap for a table later if persistence is
needed.
"""

from __future__ import annotations

import uuid

from .processor import JobSheet

_BATCHES: dict[str, list[JobSheet]] = {}


def create_batch(sheets: list[JobSheet]) -> str:
    batch_id = uuid.uuid4().hex[:12]
    _BATCHES[batch_id] = sheets
    return batch_id


def get_batch(batch_id: str) -> list[JobSheet] | None:
    return _BATCHES.get(batch_id)


def save_batch(batch_id: str, sheets: list[JobSheet]) -> None:
    _BATCHES[batch_id] = sheets


def delete_batch(batch_id: str) -> None:
    _BATCHES.pop(batch_id, None)
