# Contributing to SparkleFrame

These guides are the detailed companions to the top-level [AGENTS.md](../../AGENTS.md) (the lean
entry point) and [ARCHITECTURE.md](../../ARCHITECTURE.md) (the codemap). Read the guide for the
area you're working in before you start.

## The guides

| Guide | Read it when… |
|---|---|
| [adding-a-feature.md](adding-a-feature.md) | You're adding or changing a `functions` / `column` / `dataframe` API — the end-to-end checklist. |
| [testing.md](testing.md) | You're writing tests — parity tests, valid + malformed coverage, the oracle. |
| [spark4-semantics.md](spark4-semantics.md) | You're touching cast/parse or any strict-vs-lenient (`try_*`) behavior. |
| [engines.md](engines.md) | You're working on an engine or the parity gate / ratchet. |

## The contribution loop at a glance

1. **Locate** the right layer — engine package facade (`sparkleframe/<engine>/`) + its
   `*_helpers.py`. See [ARCHITECTURE.md](../../ARCHITECTURE.md).
2. **Implement** following the engine conventions; for cast/parse, follow the strict/lenient
   pattern in [spark4-semantics.md](spark4-semantics.md).
3. **Test** against real Spark with parity tests covering valid **and** malformed inputs
   ([testing.md](testing.md)).
4. **Update the gate** if you promoted a behavior on an engine ([engines.md](engines.md)).
5. **Verify** with `make pr-check` (black + lint + coverage), always in Docker.

The full version is in [adding-a-feature.md](adding-a-feature.md).
