# Figures

Two architecture scenes live here:

| Scene | Render | Claim |
| --- | --- | --- |
| `compilation.excalidraw` | `compilation.png` | A PyTorch graph becomes MLIR, partitions, and backend artifacts |
| `runtime.excalidraw` | `runtime.png` | ForgeRT schedules a plan and dispatches to providers |

`generate.py` writes both scenes with fixed seeds and geometry. Run it from the
repository root after a layout change:

```bash
python3 docs/figures/generate.py
```

Render with the Excalidraw CLI:

```bash
excalidraw render docs/figures/compilation.excalidraw \
  --out docs/figures/compilation.png \
  --scale 2 --max-width 3200

excalidraw render docs/figures/runtime.excalidraw \
  --out docs/figures/runtime.png \
  --scale 2 --max-width 3200
```

The CLI needs Chrome, Chromium, Edge, or Firefox. Set
`EXCALIDRAW_BROWSER_PATH` to the browser binary when it is not on the default
path.

Inspect both PNGs after a render:

- Each figure makes one claim.
- Every label fits inside its box.
- Every arrow has a label and crosses no text.
- The smallest text is readable at page width.
- The scene and the PNG are both current.

The palette comes from `agents/excalidraw.nix` in the personal nixos
repository. Strokes carry the role, fills stay pale.
