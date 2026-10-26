"""Shared fixtures: the synthetic site, built once, and each case run at most once.

    pytest                   # compare every case against tests/golden/
    pytest --update-golden   # rewrite tests/golden/ from the current code

Only update the golden files when an output change is intended, and review
the git diff of tests/golden/ before committing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import cases  # noqa: E402
import synth  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden", action="store_true",
        help="Rewrite tests/golden/ from the current output instead of comparing against it",
    )


@pytest.fixture(scope="session")
def update_golden(request) -> bool:
    return request.config.getoption("--update-golden")


@pytest.fixture(scope="session")
def site(tmp_path_factory) -> synth.Site:
    return synth.build(tmp_path_factory.mktemp("site"))


@pytest.fixture(scope="session")
def results(site, tmp_path_factory):
    """Lazily run cases by name; each runs once per session."""
    done: dict[str, cases.Result] = {}
    root = tmp_path_factory.mktemp("out")

    def get(name: str) -> cases.Result:
        if name not in done:
            done[name] = cases.run(cases.BY_NAME[name], site, root / name)
        return done[name]

    return get
