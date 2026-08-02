# TODO(asset): bundled monospace font

No font file is bundled here yet. `ui.theme.load_monospace_font()` looks
for a `.ttf`/`.otf` in this directory and, finding none, falls back to an
installed system monospace family (logging a warning either way it falls
back) -- so the application runs correctly without this, just without the
guarantee that the layout matches the mockup pixel-for-pixel on a machine
that does not happen to have JetBrains Mono or IBM Plex Mono installed.

Before shipping to the competition laptop: download either family's
OFL-licensed `.ttf` (JetBrains Mono, IBM Plex Mono) into this directory.
Nothing else needs to change -- `load_monospace_font()` picks up any
`.ttf`/`.otf` placed here automatically.
