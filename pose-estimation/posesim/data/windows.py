"""Non-owning references to fixed windows within whole segments."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class WindowRef:
    segment_index: int
    start: int
    stop: int

    def __post_init__(self):
        if self.segment_index < 0 or self.start < 0 or self.stop <= self.start:
            raise ValueError("window reference indices must define a non-empty range")


def window_refs(segments, size=64, stride=32):
    """Return references into concatenated whole segments without copying samples."""
    if size <= 0 or stride <= 0:
        raise ValueError("size and stride must be positive")
    refs = []
    cursor = 0
    for segment_index, segment in enumerate(segments):
        if isinstance(segment, tuple) and len(segment) == 2:
            start, stop = map(int, segment)
        else:
            start, stop = cursor, cursor + len(segment)
        if start != cursor or stop < start:
            raise ValueError("segment ranges must be contiguous and ordered")
        refs.extend(WindowRef(segment_index, offset, offset + size)
                    for offset in range(start, stop - size + 1, stride))
        cursor = stop
    return tuple(refs)
