"""Microbenchmark: pythondf (pure Python) vs polarsdf (Polars-backed).

Measures per-op latency at row counts spanning the low-latency regime
(1 row) through bulk processing (1M rows). Reports median time across
N iterations chosen per row count.

Run:
    python scripts/bench_pythondf_vs_polarsdf.py
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

from sparkleframe.polarsdf import functions as PF
from sparkleframe.polarsdf.session import SparkSession as PolarsSession
from sparkleframe.polarsdf.window import Window as PolarsWindow
from sparkleframe.pythondf import functions as YF
from sparkleframe.pythondf.session import SparkSession as PythonSession
from sparkleframe.pythondf.window import Window as PythonWindow

ROW_COUNTS = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]


def iters_for(n: int) -> int:
    if n <= 10:
        return 5_000
    if n <= 100:
        return 2_000
    if n <= 1_000:
        return 500
    if n <= 10_000:
        return 100
    if n <= 100_000:
        return 30
    return 10


def median_seconds(fn: Callable[[], object], iters: int) -> float:
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def make_data(n: int) -> list[dict]:
    # Row-oriented input — realistic for low-latency feature serving — accepted by both backends.
    return [{"a": i, "b": i + n} for i in range(n)]


def make_grouped_data(n: int) -> list[dict]:
    # Same shape + a low-cardinality grouping key so groupBy/window have ~10 groups.
    return [{"a": i, "b": i + n, "k": i % 10} for i in range(n)]


def make_string_data(n: int) -> list[dict]:
    import string
    alphabet = string.ascii_uppercase
    return [{"s": alphabet[i % 26] + alphabet[(i // 26) % 26]} for i in range(n)]


def make_join_pair(n: int):
    left = [{"k": i, "a": i} for i in range(n)]
    right = [{"k": i, "b": i * 2} for i in range(n)]
    return left, right


# --- Workloads (each takes a backend's session + data and runs one op end-to-end) ---


def bench_create(session, F_mod, data):  # noqa: ARG001
    return session.createDataFrame(data)


def bench_with_column(session, F_mod, df_factory):
    df = df_factory()
    return df.withColumn("c", F_mod.col("a") + F_mod.col("b"))


def bench_filter(session, F_mod, df_factory):
    df = df_factory()
    return df.filter(F_mod.col("a") > F_mod.lit(0))


def bench_pipeline(session, F_mod, df_factory):
    """Realistic chained pipeline: 2 withColumn + filter + select."""
    df = df_factory()
    return (
        df.withColumn("c", F_mod.col("a") + F_mod.col("b"))
        .withColumn("d", F_mod.col("c") * F_mod.lit(2))
        .filter(F_mod.col("a") >= F_mod.lit(0))
        .select("a", "d")
    )


def bench_groupby_sum(session, F_mod, df_factory):
    df = df_factory()
    return df.groupBy("k").agg(F_mod.sum("a"))


def bench_orderby(session, F_mod, df_factory):
    df = df_factory()
    return df.orderBy(F_mod.col("a").desc())


def bench_join_inner(session, F_mod, dfs):
    left, right = dfs
    return left.join(right, on="k", how="inner")


def bench_string_funcs(session, F_mod, df_factory):
    # Uses lower+concat+rlike (polarsdf has no F.upper).
    df = df_factory()
    return df.withColumn(
        "s2", F_mod.concat(F_mod.lower(F_mod.col("s")), F_mod.lit("_"), F_mod.col("s"))
    ).filter(F_mod.col("s2").rlike("^a"))


def bench_window_running_sum(session, F_mod, df_factory):
    df = df_factory()
    Window = PolarsWindow if F_mod is PF else PythonWindow
    spec = Window.partitionBy("k").orderBy("a")
    return df.withColumn("running", F_mod.sum("a").over(spec))


WORKLOADS = [
    # (label, fn, needs_df, build_extra_kind)
    # build_extra_kind: None=use main data, "grouped"=add "k" col, "two_df"=build a second df for join,
    # "string"=add a string column.
    ("createDataFrame", bench_create, False, None),
    ("withColumn(a+b)", bench_with_column, True, None),
    ("filter(a>0)", bench_filter, True, None),
    ("pipeline (2wc+filter+select)", bench_pipeline, True, None),
    ("groupBy(k).sum(a)", bench_groupby_sum, True, "grouped"),
    ("orderBy(a desc)", bench_orderby, True, None),
    ("join inner on=k", bench_join_inner, True, "two_df"),
    ("string concat+lower+rlike", bench_string_funcs, True, "string"),
    # Window: polarsdf Column has no .over() — pythondf-only, no parity comparison.
]


def run() -> None:
    py_session = PythonSession()
    pl_session = PolarsSession()

    print(f"{'workload':<40} {'rows':>10} {'pythondf':>14} {'polarsdf':>14} {'speedup':>14}")
    print("-" * 96)

    for label, workload, takes_df, extra in WORKLOADS:
        for n in ROW_COUNTS:
            iters = iters_for(n)

            if not takes_df:
                data = make_data(n)
                py_t = median_seconds(lambda: workload(py_session, YF, data), iters)
                pl_t = median_seconds(lambda: workload(pl_session, PF, data), iters)
            elif extra == "two_df":
                # Cap n at 10k for join — quadratic worst-case otherwise.
                if n > 10_000:
                    print(f"{label:<40} {n:>10,} {'skipped (>10k)':>14}")
                    continue
                left, right = make_join_pair(n)
                py_left = py_session.createDataFrame(left)
                py_right = py_session.createDataFrame(right)
                pl_left = pl_session.createDataFrame(left)
                pl_right = pl_session.createDataFrame(right)
                py_t = median_seconds(lambda: workload(py_session, YF, (py_left, py_right)), iters)
                pl_t = median_seconds(lambda: workload(pl_session, PF, (pl_left, pl_right)), iters)
            else:
                if extra == "grouped":
                    data = make_grouped_data(n)
                elif extra == "string":
                    data = make_string_data(n)
                else:
                    data = make_data(n)
                py_df = py_session.createDataFrame(data)
                pl_df = pl_session.createDataFrame(data)
                py_t = median_seconds(lambda: workload(py_session, YF, lambda d=py_df: d), iters)
                pl_t = median_seconds(lambda: workload(pl_session, PF, lambda d=pl_df: d), iters)

            speedup = pl_t / py_t if py_t > 0 else float("inf")
            winner = "py" if speedup > 1 else "pl"
            arrow = f"{speedup:>7.2f}x {winner}"
            print(f"{label:<40} {n:>10,} {fmt(py_t):>14} {fmt(pl_t):>14} {arrow:>14}")
        print()


def fmt(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


if __name__ == "__main__":
    run()
