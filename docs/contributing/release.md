# Releasing SparkleFrame

Releases are fully automated by GitHub Actions. Releasing a new version means running one
workflow from the Actions tab. That workflow publishes the wheel to PyPI and the docs to GitHub
Pages. You never pick the version number by hand; it is calculated from the commit history.

## How versioning works

- **Version source of truth:** `release/X.Y.Z` branches on `origin`, not git tags. The latest
  release is the highest `origin/release/*` branch.
- **Next version:** [scripts/calculate_version.py](../../scripts/calculate_version.py) reads every
  non-merge commit between the latest `release/X.Y.Z` branch and the branch being released, and
  uses semver driven by the commit-type prefix of each commit's first line:

  | Any commit summary looks like | Bump | Example |
  |---|---|---|
  | `<type>!: ...` (e.g. `feat!:`, `fix!:`) | **major** | `1.3.0 → 2.0.0` |
  | `feat: ...` / `feat(scope): ...` | **minor** | `1.3.0 → 1.4.0` |
  | anything else (`fix:`, `chore:`, `test:` ...) | **patch** | `1.3.0 → 1.3.1` |

  If there are no new commits since the last release branch, the run aborts with
  `Release would be made without any commits`.
- **Changelog:** [scripts/generate_changelog.py](../../scripts/generate_changelog.py) collects GitHub
  issue references (`#123`) from those commit summaries, looks up each issue's title, and adds a
  new `release/X.Y.Z` section on top of the previous release's `changelog.md`. **A commit with no
  `#issue` in its summary doesn't appear in the changelog.**
- **Package version:** `flit` reads `__version__` from `sparkleframe/__init__.py`, which loads
  [sparkleframe/version.txt](../../sparkleframe/version.txt). CI writes that file during the release;
  don't edit it by hand.

## Before you release

1. Merge everything that should ship into `main` and make sure `verification` is green on it.
2. Check the commit summaries since the last release follow the `<type>: ... (#issue)` convention,
   so the version bump and changelog come out right. To see what's going in:
   ```bash
   git fetch origin
   git log --no-merges --format='%s' "$(git for-each-ref --format='%(refname:short)' 'refs/remotes/origin/release/*' | sort -V | tail -1)"..origin/main
   ```
3. (Optional) Preview the version locally. This runs on your host, not in Docker, because the
   container doesn't mount `.git` or `scripts/`. You need Python ≥ 3.11 with the dev
   requirements (at least `requests`) installed, e.g. in a venv:
   ```bash
   python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
   source .venv/bin/activate
   ```
   Then:
   ```bash
   python scripts/calculate_version.py origin/main
   git checkout sparkleframe/version.txt   # the script overwrites this file
   ```
   To preview the changelog too, run `GITHUB_TOKEN=<token> python scripts/generate_changelog.py origin/main`
   (it writes `changelog.md`; discard it afterwards).

## Release steps

### 1. Dry run (recommended): `prepare-deployment`

Actions → **prepare-deployment** → *Run workflow* on `main`.

It runs the full `verification` workflow (black, ruff, script tests, coverage) and then builds,
without publishing anything:

- `version.txt`: the version that would be released
- `changelog.md`: the changelog that would be published
- `supported_api_doc.md`: the regenerated PySpark API coverage page
- `sparkleframe-X.Y.Z-py3-none-any.whl`: the wheel

Download the artifacts and check them. You can `pip install` the wheel into a scratch env to test
it.

### 2. Release: `deploy-docs-pypi`

Actions → **deploy-docs-pypi** → *Run workflow* on `main`.

The workflow:

1. Re-runs `prepare-deployment` (verification + version/changelog/API-doc/wheel).
2. Creates the branch `release/X.Y.Z` from the commit it ran on, commits `sparkleframe/version.txt`,
   `changelog.md` and `docs/supported_api_doc.md` to it, and pushes it. **This branch becomes the
   baseline for the next release.**
3. Publishes to PyPI with `flit publish`, using the `PYPI_API_TOKEN` repo secret.
4. Deploys the docs with `mike` (`make docs-deploy version=X.Y.Z`) under the `X.Y` alias, and
   moves `latest` to point at it.

### 3. Verify

- `pip install sparkleframe==X.Y.Z` works, and https://pypi.org/project/sparkleframe/ shows the new
  version.
- The docs site shows the new version as `latest`, including the changelog.
- `origin/release/X.Y.Z` exists.

## Required configuration

| Secret | Used for |
|---|---|
| `PYPI_API_TOKEN` | `flit publish` (username `__token__`) |
| `GITHUB_TOKEN` | provided automatically; changelog issue lookups, pushing the release branch and `gh-pages` |

## Troubleshooting

- **`Release would be made without any commits`:** nothing new since the last `release/*`
  branch. Merge something first.
- **The bump is wrong (e.g. patch when you expected minor):** a commit summary doesn't start with
  `feat:` / `feat(scope):`. The parser matches the first `word:` in the summary. Fix it with
  another commit using the correct prefix. Don't rewrite history on `main`.
- **Changelog is missing entries:** the commit summaries had no `#issue` reference.
- **Workflow failed after pushing `release/X.Y.Z` but before PyPI upload:** the branch now exists,
  so a re-run would treat `X.Y.Z` as already released and either bump past it or abort with "no
  commits". Delete the branch (`git push origin --delete release/X.Y.Z`), then re-run
  `deploy-docs-pypi`.
- **PyPI upload succeeded but docs deploy failed:** fix the cause, then deploy the docs alone from
  the release branch:
  ```bash
  git checkout release/X.Y.Z
  make docs-deploy version=X.Y.Z
  ```
- **A bad version reached PyPI:** PyPI never lets you re-upload the same version. *Yank* it in the
  PyPI UI, merge the fix, and release again (it ships as the next patch).

## Local build (no publishing)

```bash
make wheel        # flit build --format=wheel → dist/
```
