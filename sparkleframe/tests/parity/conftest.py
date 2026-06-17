"""Fixtures and gating for the shared parity suite.

The ``engine`` fixture parametrizes every parity test over all runtime engines.
A test written once therefore runs as ``...[POLARS]`` and ``...[PYTHON]``.
Engines that are not scaffolded yet skip; behaviors an engine does not support
yet ``xfail`` via the per-engine gate in ``gaps.py``.
"""

from __future__ import annotations

import pytest

from sparkleframe.tests.parity.engines import ENGINES
from sparkleframe.tests.parity.gaps import GATES


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "feature(name): the Spark behavior this parity test exercises; gated per engine in gaps.py",
    )


@pytest.fixture(params=list(ENGINES.keys()), ids=lambda e: e.name)
def engine(request):
    adapter = ENGINES[request.param]
    if adapter is None:
        pytest.skip(f"{request.param.name} engine not scaffolded yet")
    return adapter


def pytest_collection_modifyitems(config, items):
    """xfail each parity test whose feature is gated for the engine it runs on."""
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        engine_param = callspec.params.get("engine")
        if engine_param is None:
            continue
        # GATES is keyed by lower-case engine name (e.g. "polars") — see gaps.py.
        gate_key = engine_param.name.lower()
        marker = item.get_closest_marker("feature")
        if marker is None or not marker.args:
            continue
        feature = marker.args[0]
        if feature in GATES.get(gate_key, set()):
            item.add_marker(
                pytest.mark.xfail(
                    reason=f"{engine_param.name} engine: '{feature}' not supported yet",
                    strict=False,
                )
            )
