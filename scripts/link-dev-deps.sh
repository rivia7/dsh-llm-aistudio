#!/bin/sh
# Link the dsh peer packages from the global dsh install into ./node_modules so
# unit tests and the live smoke script can import them outside dsh. Inside dsh
# the launcher routes these bare imports itself, so nothing here is installed.
set -eu
DSH_MODULES="${DSH_MODULES:-$(npm root -g)/@deepseek-ai/dsh/node_modules}"
cd "$(dirname "$0")/.."
mkdir -p node_modules/@deepseek-ai
for pkg in dsh-llm schemastery dsh-credentials dsh-launch-environment cordis; do
  src="$DSH_MODULES/@deepseek-ai/$pkg"
  [ -d "$src" ] || { echo "missing $src" >&2; exit 1; }
  ln -sfn "$src" "node_modules/@deepseek-ai/$pkg"
done
# yaml reads the dsh credential store in the live smoke script.
ln -sfn "$DSH_MODULES/yaml" node_modules/yaml
echo "linked dev deps from $DSH_MODULES"
