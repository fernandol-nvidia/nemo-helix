# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read an ATIF file one step at a time.

A long agent run records gigabytes of ATIF, mostly per-step LLM requests and the producer's event log.
Parsed whole, Python holds that text at up to four bytes per character. :func:`read_atif_lazily`
keeps each step as its raw JSON and validates it on access, and never builds the top-level ``extra``.
"""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO, Any, overload

from nhx_evals_sdk.values.atif import Step, Trajectory

_CHUNK_CHARS = 16 * 1024 * 1024
_DECODER = json.JSONDecoder()
_WHITESPACE = re.compile(r"[ \t\n\r]*")
_NUMBER_CHARS = frozenset("0123456789.eE+-")
# Containers are walked member by member down to this depth; deeper values are decoded whole.
_SKIP_DEPTH = 2


class _RawSteps(Sequence[Step]):
    """Steps kept as their raw JSON and validated on each access."""

    def __init__(self, raw: list[bytes]) -> None:
        self._raw = raw

    def __len__(self) -> int:
        return len(self._raw)

    @overload
    def __getitem__(self, index: int) -> Step: ...

    @overload
    def __getitem__(self, index: slice) -> list[Step]: ...

    def __getitem__(self, index: int | slice) -> Step | list[Step]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        return Step.model_validate_json(self._raw[index])


class _Scanner:
    """Decode JSON values one at a time from a text stream, holding only the unread part."""

    def __init__(self, handle: IO[bytes]) -> None:
        self._handle = handle
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._text = ""
        self._position = 0
        self._ended = False

    def _read_more(self) -> bool:
        if self._ended:
            return False
        chunk = self._handle.read(_CHUNK_CHARS)
        self._ended = not chunk
        self._text = self._text[self._position :] + self._decoder.decode(chunk, final=self._ended)
        self._position = 0
        return True

    def peek(self) -> str:
        """Return the next non-whitespace character without consuming it."""
        while True:
            self._position = _WHITESPACE.match(self._text, self._position).end()  # type: ignore[union-attr]
            if self._position < len(self._text):
                return self._text[self._position]
            if not self._read_more():
                raise ValueError("ATIF document ends early")

    def take(self, expected: str) -> None:
        if self.peek() != expected:
            raise ValueError(f"ATIF document: expected {expected!r} at offset {self._position}")
        self._position += 1

    def take_separator(self, closer: str) -> bool:
        """Consume a comma and return True, or consume ``closer`` and return False."""
        if self.peek() == ",":
            self._position += 1
            return True
        self.take(closer)
        return False

    def value(self) -> tuple[Any, str]:
        """Decode the next value and return it with its raw JSON text."""
        self.peek()
        while True:
            try:
                decoded, end = _DECODER.raw_decode(self._text, self._position)
            except json.JSONDecodeError:
                if self._read_more():
                    continue
                raise
            # A number may continue in the next chunk: "89." decodes as 89 until the fraction arrives.
            number = isinstance(decoded, int | float)
            if number and (end == len(self._text) or self._text[end] in _NUMBER_CHARS) and self._read_more():
                continue
            raw = self._text[self._position : end]
            self._position = end
            return decoded, raw

    def members(self) -> Iterator[str]:
        """Consume an object, yielding each key once the scanner is positioned at its value."""
        self.take("{")
        if self.peek() == "}":
            self._position += 1
            return
        while True:
            key, _ = self.value()
            self.take(":")
            yield key
            if not self.take_separator("}"):
                return

    def elements(self) -> Iterator[str]:
        """Consume an array, yielding each element's raw JSON text."""
        self.take("[")
        if self.peek() == "]":
            self._position += 1
            return
        while True:
            yield self.value()[1]
            if not self.take_separator("]"):
                return

    def skip(self, depth: int = 0) -> None:
        """Consume the next value without building the containers above ``_SKIP_DEPTH``."""
        opener = self.peek()
        if opener not in "[{" or depth >= _SKIP_DEPTH:
            self.value()
            return
        if opener == "{":
            for _ in self.members():
                self.skip(depth + 1)
            return
        self.take("[")
        if self.peek() == "]":
            self._position += 1
            return
        while True:
            self.skip(depth + 1)
            if not self.take_separator("]"):
                return


def read_atif_lazily(path: Path) -> Trajectory:
    """Read the ATIF trajectory at ``path`` with each step validated on access and no top-level ``extra``.

    The header and first step are validated as :class:`Trajectory` on read. Readers that need the
    trajectory-level ``extra`` use :class:`~nhx_evals_sdk.values.evidence.ATIFTraceHandle.trace`.
    """
    header: dict[str, Any] = {}
    raw_steps: list[bytes] = []
    with path.open("rb") as handle:
        scanner = _Scanner(handle)
        for key in scanner.members():
            if key == "steps":
                raw_steps.extend(raw.encode() for raw in scanner.elements())
            elif key == "extra":
                scanner.skip()
            else:
                header[key], _ = scanner.value()
    first = [json.loads(raw_steps[0])] if raw_steps else []
    trajectory = Trajectory.model_validate({**header, "steps": first})
    return trajectory.model_copy(update={"steps": _RawSteps(raw_steps)})
