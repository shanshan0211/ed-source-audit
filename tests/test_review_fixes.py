"""Unit tests for the review fixes: audit.stats.bh_fdr (SPEC-style, standalone).

Run directly:
    python tests/test_review_fixes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wjem_ed_audit.audit.stats import bh_fdr  # noqa: E402


def test_bh_fdr_monotone_in_raw_p():
    """Adjusted p must be monotone non-decreasing in the raw p rank order."""
    p = np.array([0.001, 0.01, 0.02, 0.03, 0.5])
    q = bh_fdr(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12), f"not monotone: {q[order]}"


def test_bh_fdr_bounds_and_floor():
    """q_i >= p_i elementwise, all in [0, 1]; smallest p gets m*penalty."""
    p = np.array([0.04, 0.001, 0.9])
    q = bh_fdr(p)
    assert np.all(q >= p - 1e-12) and np.all((q >= 0) & (q <= 1)), q
    assert np.isclose(q[1], 0.001 * 3), q  # rank-1 of m=3 -> p*m/1


def test_bh_fdr_nan_and_all_null():
    """NaN passes through; the surviving family of m=2 adjusts to [0.6, 0.7]."""
    p = np.array([0.3, np.nan, 0.7])
    q = bh_fdr(p)
    assert np.isnan(q[1]) and np.isclose(q[0], 0.6) and np.isclose(q[2], 0.7), q


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}")
    print(f"\n{len(fns) - failed}/{len(fns)} review-fix tests passed")
    sys.exit(1 if failed else 0)
