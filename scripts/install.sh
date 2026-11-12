#!/bin/sh
# Pack this plugin and (re)install it into a dsh profile (default: web).
#   sh scripts/install.sh [profile]
# Restart `dsh <profile>` afterwards to load the new version.
set -eu
PROFILE="${1:-web}"
cd "$(dirname "$0")/.."
VERSION=$(node -p "require('./package.json').version")
mkdir -p dist
rm -f "dist/dsh-llm-aistudio-$VERSION.tgz"
npm pack --pack-destination dist >/dev/null
# `dsh plugin` runs the first pnpm on PATH. Use the terminal's pnpm (~/Library/pnpm,
# store v11) when present, so the profile's node_modules stays on one store.
if [ -x "$HOME/Library/pnpm/pnpm" ]; then PATH="$HOME/Library/pnpm:$PATH"; export PATH; fi
dsh plugin --profile "$PROFILE" add "file:$PWD/dist/dsh-llm-aistudio-$VERSION.tgz"
echo "installed dsh-llm-aistudio $VERSION into profile $PROFILE; restart dsh $PROFILE to load it"
