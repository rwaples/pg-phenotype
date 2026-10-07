#!/usr/bin/env bash
# Build the R source tarball of pgphenotype, the only producer of one: the
# checked-in r/ reaches the core through ../../../crates/core, which a tarball
# cannot, so this stages a copy.
#
#   tools/r_build_tarball.sh [out-dir]       # default: target/r
#
# In a temporary copy of r/ it stages crates/core at src/rust/core with the
# workspace-inherited keys and dependencies written out (pedigree-graph-core
# as its git+rev pin), points the binding crate at it, vendors the complete
# locked dependency graph, git sources included, into src/rust/vendor.tar.xz
# (with the source-replacement config Makevars installs), lists every vendored
# crate's authors and license in inst/AUTHORS, and runs R CMD build.
# Needs cargo, python3 and R on PATH (pixi run -e r tools/r_build_tarball.sh).
set -euo pipefail

# Its vendored build ignores PG_PHENOTYPE_CARGO_FEATURES anyway; refusing keeps a
# dev shell's test-hooks setting from being mistaken for a supported build.
if [ -n "${PG_PHENOTYPE_CARGO_FEATURES:-}" ]; then
  echo "r_build_tarball.sh: unset PG_PHENOTYPE_CARGO_FEATURES (=$PG_PHENOTYPE_CARGO_FEATURES); the tarball builds without features" >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(realpath -m "${1:-$REPO/target/r}")"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
PKG="$STAGE/pgphenotype"

# The package, without build products.
mkdir -p "$PKG"
tar -C "$REPO/r" --exclude=src/rust/target --exclude='src/*.o' --exclude='src/*.so' \
  --exclude=src/.cargo -cf - . | tar -C "$PKG" -xf -

# The core: its library sources only, with what the workspace supplied
# written out.
CORE="$PKG/src/rust/core"
mkdir -p "$CORE"
cp -R "$REPO/crates/core/src" "$CORE/src"
python3 - "$REPO/Cargo.toml" "$REPO/crates/core/Cargo.toml" "$CORE/Cargo.toml" \
  "$PKG/src/rust/Cargo.toml" "$PKG/DESCRIPTION" <<'PY'
import re
import sys
import tomllib

root_path, core_in, core_out, binding_path, description_path = sys.argv[1:]
with open(root_path, "rb") as f:
    workspace = tomllib.load(f)["workspace"]
package, dependencies = workspace["package"], workspace["dependencies"]


def inline(value):
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(inline(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {inline(v)}" for k, v in value.items()) + " }"
    raise SystemExit(f"r_build_tarball.sh: cannot write {value!r} inline")


section = None
lines = []
for line in open(core_in).read().splitlines():
    header = re.match(r"^\[(.+)\]$", line)
    if header:
        section = header.group(1)
    inherited = re.match(r"^([A-Za-z0-9_-]+)\.workspace = true$", line)
    if inherited:
        key = inherited.group(1)
        table = package if section == "package" else dependencies
        line = f"{key} = {inline(table[key])}"
    lines.append(line)
text = "\n".join(lines) + "\n"
if re.search(r"workspace\s*=\s*true", text):
    raise SystemExit("r_build_tarball.sh: an inherited key is left in the staged core Cargo.toml")
open(core_out, "w").write(text)

with open(binding_path, "rb") as f:
    binding_version = tomllib.load(f)["package"]["version"]
description_version = re.search(r"^Version: (\S+)$", open(description_path).read(), re.M).group(1)
versions = {"workspace": package["version"], "r/src/rust": binding_version,
            "r/DESCRIPTION": description_version}
if len(set(versions.values())) != 1:
    raise SystemExit(f"r_build_tarball.sh: versions disagree: {versions}")
PY
sed -i 's|path = "../../../crates/core"|path = "core"|' "$PKG/src/rust/Cargo.toml"

# Vendor exactly the locked graph, then compress it.  The config cargo prints
# replaces crates-io and the pedigree-graph git source alike.
(
  cd "$PKG/src/rust"
  cargo vendor --locked --versioned-dirs vendor > vendor-config.toml
  tar --owner=0 --group=0 --numeric-owner -cJf vendor.tar.xz vendor
  rm -rf vendor
)

# Authors and licenses of every vendored crate, for DESCRIPTION's Copyright.
mkdir -p "$PKG/inst"
{
  echo "The pgphenotype source tarball bundles these Rust crates (src/rust/vendor.tar.xz)."
  echo "Each is listed with its version, license and authors as its Cargo manifest states"
  echo "(or, where the manifest names none, its repository)."
  echo
  cargo metadata --locked --format-version 1 --manifest-path "$PKG/src/rust/Cargo.toml" |
    python3 -c '
import json, sys
meta = json.load(sys.stdin)
for p in sorted(meta["packages"], key=lambda p: (p["name"], p["version"])):
    if p["source"] is None:
        continue
    name, version = p["name"], p["version"]
    license = p["license"] or "see the crate"
    # A manifest may omit authors; its repository then names the holders.
    authors = ", ".join(p["authors"]) or "the authors of " + (p["repository"] or name)
    print(f"{name} {version} ({license}): {authors}")
'
} > "$PKG/inst/AUTHORS"

mkdir -p "$OUT"
( cd "$OUT" && R CMD build --no-manual "$PKG" )
ls -l "$OUT"/pgphenotype_*.tar.gz
