#!/usr/bin/env bash
# Fetch lettering fonts that are actually legal to redistribute.
#
# What is NOT here, and why:
#
#   Wild Words        Comicraft, commercial. The face scanlate.io offers.
#                     Not redistributable. Buy it if you want it.
#   Anime Ace         Blambot. Free for personal / non-commercial indie use
#                     only, trademarked. Cannot be bundled in a product or
#                     used for for-profit work without a licence.
#   Badaboom, etc.    Blambot, same terms.
#   Komika            Freeware, but terms vary per weight. Treat as
#                     user-supplied rather than bundling it.
#
# Bundling any of the above would be its own infringement, entirely separate
# from whatever manga you point this tool at. So: OFL and Apache only.
#
#   ./scripts/fetch-fonts.sh [target-dir]

set -euo pipefail

readonly DEST="${1:-$HOME/.local/share/mangatl/fonts}"
mkdir -p "$DEST"

# name|licence|url
readonly FONTS=(
  "ComicNeue-Bold.ttf|OFL-1.1|https://github.com/google/fonts/raw/main/ofl/comicneue/ComicNeue-Bold.ttf"
  "ComicNeue-BoldItalic.ttf|OFL-1.1|https://github.com/google/fonts/raw/main/ofl/comicneue/ComicNeue-BoldItalic.ttf"
  "ComicNeue-Regular.ttf|OFL-1.1|https://github.com/google/fonts/raw/main/ofl/comicneue/ComicNeue-Regular.ttf"
  "Bangers-Regular.ttf|Apache-2.0|https://github.com/google/fonts/raw/main/apache/bangers/Bangers-Regular.ttf"
  "PatrickHand-Regular.ttf|OFL-1.1|https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf"
)

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

for spec in "${FONTS[@]}"; do
  IFS='|' read -r name licence url <<< "$spec"
  target="$DEST/$name"
  if [[ -f "$target" ]]; then
    log "have $name ($licence)"
    continue
  fi
  log "fetching $name ($licence)"
  if ! curl -fsSL --retry 3 -o "$target.part" "$url"; then
    printf '\033[1;33m!!! \033[0mfailed: %s -- skipping\n' "$name" >&2
    rm -f "$target.part"
    continue
  fi
  mv "$target.part" "$target"
done

# Noto Sans JP is needed only if you ever render Japanese back onto a page
# (SFX annotations, bilingual proofs). OFL-1.1, fetched from the same source.
if [[ ! -f "$DEST/NotoSansJP-Bold.ttf" ]]; then
  log "fetching NotoSansJP-Bold.ttf (OFL-1.1)"
  curl -fsSL --retry 3 \
    -o "$DEST/NotoSansJP-Bold.ttf" \
    "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/JP/NotoSansJP-Bold.otf" \
    || printf '\033[1;33m!!! \033[0mNoto CJK fetch failed -- optional, continuing\n' >&2
fi

cat > "$DEST/LICENSES.md" <<'EOF'
# Fonts bundled with mangatl

Every font in this directory is redistributable under a permissive licence.

| Font | Licence | Use |
|---|---|---|
| Comic Neue (Regular/Bold/BoldItalic) | SIL OFL 1.1 | body dialogue |
| Bangers | Apache-2.0 | shouts, impact, SFX |
| Patrick Hand | SIL OFL 1.1 | handwriting, margin notes |
| Noto Sans JP Bold | SIL OFL 1.1 | Japanese fallback |

Deliberately absent: Wild Words (Comicraft, commercial), the Blambot
families including Anime Ace (free for personal/indie non-commercial use
only, trademarked), and Komika (freeware with per-weight terms).

You may drop any of those into this directory yourself and point
`MANGATL_FONT_DIR` at it. mangatl will use whatever it finds. Doing so is
your licensing decision, not one this project makes for you.
EOF

log "done -- $(find "$DEST" -name '*.ttf' -o -name '*.otf' | wc -l) font files in $DEST"
