#!/bin/sh
# Build the documentation site into site/.
# Run from the repository root: uv run --group docs --extra recommended sh docs/build.sh
set -e

# Execute the notebooks and convert them to the Markdown pages the nav references. They
# run in parallel, one process per notebook, to stay within the 15-minute limit of the
# Read the Docs builds: most of their time is spent compiling the spherical harmonic
# transforms, on a single core. At most 4 processes, to bound the memory.
jobs=$(nproc)
[ "$jobs" -gt 4 ] && jobs=4
echo "Executing the notebooks with $jobs processes"
printf '%s\n' docs/user-guide/*.ipynb | xargs -P "$jobs" -n 1 jupyter nbconvert --to markdown --execute

zensical build

# The theme pins the navigation sidebar only from 76.25em (1220px) up and uses a
# hamburger drawer below; lower that breakpoint to 60em (960px) so the desktop
# layout, with the navigation always visible on the left, applies on laptop-sized
# windows too. The values are hardcoded in the minified assets, shared by the CSS
# media queries and the JS matchMedia observers.
sed -i 's/76\.25em/60em/g; s/76\.234375em/59\.984375em/g' \
    site/assets/stylesheets/*/main.*.min.css \
    site/assets/javascripts/bundle.*.min.js
