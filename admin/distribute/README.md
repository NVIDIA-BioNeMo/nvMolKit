# Pip wheel release

PyPI hosts wheels built against one RDKit version. The GitHub release hosts a
`+rdkit<X.Y.Z>` variant for every pair in `rdkit_build_matrix.yaml`; GitHub
Pages provides a pip index for each variant.

Do not upload the local-version variants to PyPI.

## Setup

The host needs Git, Docker, Conda, and `gh`.

```bash
conda create -y -n nvmolkit_release -c conda-forge python=3.12 pip
conda activate nvmolkit_release

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
python -m pip install -r admin/distribute/requirements-release.txt
set -euo pipefail

VERSION=$(tr -d '[:space:]' < VERSION)
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"$REPO/artifacts/v${VERSION}"}
WHEELHOUSE=${WHEELHOUSE:-"$ARTIFACT_ROOT/wheelhouse"}
BUILD_WORKTREE_ROOT=${BUILD_WORKTREE_ROOT:-"${TMPDIR:-/tmp}/nvmolkit-wheels-v${VERSION}"}
INDEX_DIR=${INDEX_DIR:-"$ARTIFACT_ROOT/simple-index"}
PAGES_WORKTREE=${PAGES_WORKTREE:-"${TMPDIR:-/tmp}/nvmolkit-pages-v${VERSION}"}
GH_REPO=${GH_REPO:-NVIDIA-BioNeMo/nvMolKit}
CANONICAL_RDKIT=2026.3.5
```

Set `CANONICAL_RDKIT` explicitly for each release. It must exist in
`rdkit_build_matrix.yaml`.

Build the container locally, or authenticate to the registry and set
`CIBW_MANYLINUX_X86_64_IMAGE` to a pinned tag.

```bash
docker build --network host \
    -f admin/container/manylinux_2_28_cuda12.Dockerfile \
    -t nvmolkit-manylinux-cuda12:local .
export CIBW_MANYLINUX_X86_64_IMAGE=nvmolkit-manylinux-cuda12:local
```

## 1. Build

Commit or stash tracked changes before building. Reruns skip completed wheels
verified for the same source commit and rebuild stale wheel outputs or incomplete
RDKit recipe caches.

```bash
WHEELHOUSE="$WHEELHOUSE" \
WORKTREE_ROOT="$BUILD_WORKTREE_ROOT" \
    bash admin/deploy/build_full_matrix.sh 8 2
```

The first argument sets the number of concurrent builds. The second sets the
parallelism requested from each build tool; it is not a hard OS thread limit.
Wheels, logs, and timings are under `$WHEELHOUSE`.

## 2. Test

`both` runs smoke tests for every discovered wheel, followed by full tests for
the selected pairs in `admin/test/full_test_subset.txt`.

```bash
bash admin/test/test_wheel_matrix.sh "$WHEELHOUSE" both

# Optional: full tests for every matrix pair.
bash admin/test/test_wheel_matrix.sh \
    "$WHEELHOUSE" full "$WHEELHOUSE/jobs/pairs.txt"
```

Tests run installed wheels outside the checkout. Logs are in
`$WHEELHOUSE/test_logs/`. The tests require a CUDA-capable host; the build does
not.

## 3. Stage PyPI wheels

```bash
mkdir -p "$WHEELHOUSE/pypi"
if compgen -G "$WHEELHOUSE/pypi/*.whl" >/dev/null; then
    echo "PyPI staging directory is not empty" >&2
    false
fi
cp "$WHEELHOUSE/rdkit${CANONICAL_RDKIT}"/py*/*.whl "$WHEELHOUSE/pypi/"

EXPECTED_CANONICAL=$(awk -v rdkit="$CANONICAL_RDKIT" \
    '$1 == rdkit { count++ } END { print count + 0 }' \
    "$WHEELHOUSE/jobs/pairs.txt")
ACTUAL_CANONICAL=$(find "$WHEELHOUSE/pypi" -name '*.whl' -type f | wc -l)
test "$EXPECTED_CANONICAL" -gt 0
test "$ACTUAL_CANONICAL" -eq "$EXPECTED_CANONICAL"

python admin/distribute/pin_wheel_rdkit.py --check \
    "$CANONICAL_RDKIT" "$WHEELHOUSE"/pypi/*.whl
twine check "$WHEELHOUSE"/pypi/*.whl
```

## 4. Create variants

```bash
VARIANTS_DIR="$WHEELHOUSE/variants"
test ! -e "$VARIANTS_DIR"
mkdir -p "$VARIANTS_DIR"
shopt -s nullglob

while read -r rdkit py; do
    src=("$WHEELHOUSE/rdkit${rdkit}/py${py}"/*.whl)
    test "${#src[@]}" -eq 1
    out="$VARIANTS_DIR/rdkit${rdkit}"
    mkdir -p "$out"
    python admin/distribute/retag_wheel.py \
        "${src[0]}" "$out" "rdkit${rdkit}"
done < "$WHEELHOUSE/jobs/pairs.txt"

EXPECTED_WHEELS=$(wc -l < "$WHEELHOUSE/jobs/pairs.txt")
ACTUAL_WHEELS=$(find "$VARIANTS_DIR" -name '*.whl' -type f | wc -l)
test "$ACTUAL_WHEELS" -eq "$EXPECTED_WHEELS"

for variant_dir in "$VARIANTS_DIR"/rdkit*/; do
    rdkit=$(basename "$variant_dir")
    rdkit=${rdkit#rdkit}
    python admin/distribute/pin_wheel_rdkit.py --check \
        "$rdkit" "$variant_dir"/*.whl
done
twine check "$VARIANTS_DIR"/rdkit*/*.whl
```

## 5. Generate indexes

```bash
test ! -e "$INDEX_DIR"
RELEASE_URL="https://github.com/${GH_REPO}/releases/download/v${VERSION}"
./admin/distribute/generate_simple_index.sh \
    "$VARIANTS_DIR" "$INDEX_DIR" "$RELEASE_URL"
```

## 6. Upload variants

```bash
gh release view "v${VERSION}" --repo "$GH_REPO"
find "$VARIANTS_DIR" -name '*.whl' -type f -print0 |
    xargs -0 -P 8 -n 1 gh release upload "v${VERSION}" \
        --repo "$GH_REPO" --clobber

EXPECTED_ASSETS=$(mktemp)
UPLOADED_ASSETS=$(mktemp)
find "$VARIANTS_DIR" -name '*.whl' -printf '%f\n' | sort > "$EXPECTED_ASSETS"
gh release view "v${VERSION}" --repo "$GH_REPO" --json assets \
    --jq '.assets[].name' | sort > "$UPLOADED_ASSETS"
comm -23 "$EXPECTED_ASSETS" "$UPLOADED_ASSETS"
```

`comm` must print nothing. Regenerate the indexes after replacing a release
asset because they contain wheel hashes.

## 7. Publish indexes

Merge into `docs/wheels`; replacing that directory would remove older release
links.

```bash
REMOTE=origin
git fetch "$REMOTE" github_pages_host
git worktree add -B "publish-v${VERSION}-wheels" "$PAGES_WORKTREE" \
    "$REMOTE/github_pages_host"

python admin/distribute/merge_simple_index.py \
    "$INDEX_DIR" "$PAGES_WORKTREE/docs/wheels"

git -C "$PAGES_WORKTREE" add docs/wheels
git -C "$PAGES_WORKTREE" diff --staged --check
git -C "$PAGES_WORKTREE" diff --staged --stat
git -C "$PAGES_WORKTREE" commit -m "Publish v${VERSION} variant wheel indexes"
git -C "$PAGES_WORKTREE" push "$REMOTE" HEAD:github_pages_host
git worktree remove "$PAGES_WORKTREE"
```

Preserve the indexes when replacing rendered docs:

```bash
rsync -a --delete --exclude='wheels/' <sphinx-build>/ docs/
```

## 8. Upload to PyPI

Upload last. PyPI does not permit replacing a filename.

```bash
twine check "$WHEELHOUSE"/pypi/*.whl
twine upload --repository testpypi "$WHEELHOUSE"/pypi/*.whl  # optional
twine upload "$WHEELHOUSE"/pypi/*.whl
```
