#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Every arm name written as a string literal in src/ and scripts/ must be registered.

Consumers of the record skip an arm that has no trace, so a stale name makes a
figure or table drop that arm without an error.

Only string literals are checked; comments and docstrings are not. Two kinds of
literal are skipped: f-string fragments, and a literal that is a proper prefix
of a registered arm (a stem such as ``"GP-ARD"``).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from matbo.methods import build_registry

ROOT = Path(__file__).resolve().parents[1]

#: A surrogate stem followed by at least one ``-part``.
ARM_LIKE = re.compile(r"^(TabPFN|TabICL|CatBoost|RF|GP)(-[A-Za-z0-9]+)+$")


def _string_literals(path: Path) -> Iterator[tuple[int, str]]:
    tree = ast.parse(path.read_text())
    interpolated = {
        id(part)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for part in node.values
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in interpolated
        ):
            yield node.lineno, node.value


def _sources() -> Iterator[Path]:
    for base in ("src", "scripts"):
        yield from sorted((ROOT / base).rglob("*.py"))


def test_every_arm_name_in_a_string_literal_resolves():
    known = set(build_registry().methods)
    stale = []
    for path in _sources():
        for lineno, value in _string_literals(path):
            name = value.strip()
            if not ARM_LIKE.match(name) or name in known:
                continue
            if any(a.startswith(name + "-") for a in known):
                continue
            stale.append(f"{path.relative_to(ROOT)}:{lineno}: {name!r}")
    assert not stale, "arm names that do not resolve in the registry:\n  " + "\n  ".join(stale)


def test_the_guard_would_catch_an_unregistered_arm():
    known = set(build_registry().methods)
    for gone in ("TabPFN-n8", "TabPFN-LCB2", "GP-greedy", "GP-iso"):
        assert ARM_LIKE.match(gone), f"{gone} must look like an arm to be caught"
        assert gone not in known


@pytest.mark.parametrize("live", ["TabPFN-greedy", "GP-ARD-EI", "RF-EI"])
def test_the_guard_does_not_fire_on_live_arms(live):
    assert live in set(build_registry().methods)
