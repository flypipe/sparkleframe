"""Ratchet: the Python parity gate may only shrink.

Without this, a behavior that regressed could be quietly re-added to
``PYTHON_NOT_IMPLEMENTED`` to keep CI green, instead of being fixed. The high-water
mark is a committed number that should only ever be lowered.
"""

from sparkleframe.tests.parity.gaps import PYTHON_GATE_HIGH_WATER_MARK, PYTHON_NOT_IMPLEMENTED


def test_python_gate_only_shrinks():
    assert len(PYTHON_NOT_IMPLEMENTED) <= PYTHON_GATE_HIGH_WATER_MARK, (
        "The Python parity gate grew. Implement the behavior instead of re-gating it, "
        "or if you intentionally added new parity coverage, raise the high-water mark "
        "deliberately in gaps.py."
    )


def test_high_water_mark_is_not_stale():
    # Nudge to lower the committed mark whenever the live gate has shrunk below it,
    # so the ratchet keeps biting as the engine progresses.
    assert PYTHON_GATE_HIGH_WATER_MARK == len(PYTHON_NOT_IMPLEMENTED), (
        f"Lower PYTHON_GATE_HIGH_WATER_MARK to {len(PYTHON_NOT_IMPLEMENTED)} in gaps.py "
        "to match the current gate size."
    )
