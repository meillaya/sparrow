#!/usr/bin/env python3
"""Generate the two architecture scenes for the Sparrow README.

Run from the repository root:

    python3 docs/figures/generate.py

The script writes docs/figures/compilation.excalidraw and
docs/figures/runtime.excalidraw. Render each scene with the Excalidraw CLI
after any geometry change.

The palette and defaults mirror agents/excalidraw.nix in the personal nixos
repository: blue for the compiler front end, orange for analysis, purple for
the runtime, teal for simulators, neutral for data and events.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent

PALETTE = {
    "compiler": ("#1971c2", "#d0ebff"),
    "analysis": ("#f08c00", "#fff3bf"),
    "runtime": ("#9c36b5", "#f3d9fa"),
    "model": ("#0c8599", "#c5f6fa"),
    "neutral": ("#495057", "#f1f3f5"),
}
INK = "#1e1e1e"
ARROW_INK = "#495057"
BACKGROUND = "#ffffff"
FONT_BODY = 22
FONT_TITLE = 32
FONT_NOTE = 18


def _seed(element_id: str) -> int:
    return zlib.crc32(element_id.encode()) & 0x7FFFFFFF


def _base(element_id: str, kind: str, x: float, y: float, width: float, height: float) -> dict:
    seed = _seed(element_id)
    return {
        "id": element_id,
        "type": kind,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "hachure",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 2,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": seed,
        "version": 1,
        "versionNonce": seed,
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
    }


@dataclass(frozen=True)
class Box:
    element_id: str
    x: float
    y: float
    width: float
    height: float
    label: str
    role: str


def box(b: Box) -> tuple[dict, dict]:
    stroke, fill = PALETTE[b.role]
    shape = _base(b.element_id, "rectangle", b.x, b.y, b.width, b.height)
    shape["strokeColor"] = stroke
    shape["backgroundColor"] = fill
    shape["roundness"] = {"type": 3}
    text_id = f"{b.element_id}_label"
    shape["boundElements"] = [{"type": "text", "id": text_id}]

    lines = b.label.split("\n")
    text_height = len(lines) * FONT_BODY * 1.25
    text = _base(
        text_id, "text", b.x + 10, b.y + (b.height - text_height) / 2, b.width - 20, text_height
    )
    text.update(
        {
            "strokeColor": INK,
            "text": b.label,
            "originalText": b.label,
            "fontSize": FONT_BODY,
            "fontFamily": 1,
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": b.element_id,
            "lineHeight": 1.25,
            "baseline": FONT_BODY * 0.9,
        }
    )
    return shape, text


def label(
    element_id: str, x: float, y: float, width: float, text: str, *, align: str = "center"
) -> dict:
    height = (text.count("\n") + 1) * FONT_NOTE * 1.25
    element = _base(element_id, "text", x, y, width, height)
    element.update(
        {
            "strokeColor": INK,
            "text": text,
            "originalText": text,
            "fontSize": FONT_NOTE,
            "fontFamily": 1,
            "textAlign": align,
            "verticalAlign": "top",
            "containerId": None,
            "lineHeight": 1.25,
            "baseline": FONT_NOTE * 0.9,
        }
    )
    return element


def title(text: str) -> dict:
    element = _base("title", "text", 65, 40, 1390, 45)
    element.update(
        {
            "strokeColor": INK,
            "text": text,
            "originalText": text,
            "fontSize": FONT_TITLE,
            "fontFamily": 1,
            "textAlign": "left",
            "verticalAlign": "top",
            "containerId": None,
            "lineHeight": 1.25,
            "baseline": FONT_TITLE * 0.9,
        }
    )
    return element


def arrow(element_id: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    left, top = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y1)
    element = _base(element_id, "arrow", left, top, width, height)
    element.update(
        {
            "strokeColor": ARROW_INK,
            "points": [[x1 - left, y1 - top], [x2 - left, y2 - top]],
            "lastCommittedPoint": None,
            "startBinding": None,
            "endBinding": None,
            "startArrowhead": None,
            "endArrowhead": "arrow",
            "elbowed": False,
        }
    )
    return element


def scene(elements: list[dict]) -> dict:
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": BACKGROUND,
        },
        "files": {},
    }


def compilation() -> dict:
    boxes = [
        Box("module", 65, 220, 230, 90, "PyTorch module", "compiler"),
        Box("fx", 355, 220, 230, 90, "FX graph", "compiler"),
        Box("importer", 645, 220, 230, 90, "FX importer", "compiler"),
        Box("passes", 935, 220, 230, 90, "MLIR passes", "compiler"),
        Box("artifacts", 1225, 220, 230, 90, "Backend artifacts", "neutral"),
        Box("partitioner", 355, 560, 230, 90, "Region partitioner", "analysis"),
        Box("capabilities", 645, 560, 230, 90, "Capabilities", "analysis"),
        Box("cache", 1225, 560, 230, 90, "Cache + report", "analysis"),
    ]

    elements: list[dict] = []
    for b in boxes:
        shape, text = box(b)
        elements.extend([shape, text])

    elements.extend(
        [
            arrow("a_trace", 295, 265, 355, 265),
            arrow("a_import", 585, 265, 645, 265),
            arrow("a_optimize", 875, 265, 935, 265),
            arrow("a_codegen", 1165, 265, 1225, 265),
            arrow("a_regions", 470, 310, 470, 560),
            arrow("a_score", 585, 605, 645, 605),
            arrow("a_assign", 875, 605, 1225, 310),
            arrow("a_key", 1340, 310, 1340, 560),
        ]
    )

    elements.extend(
        [
            label("l_trace", 295, 200, 60, "trace"),
            label("l_import", 585, 200, 60, "import"),
            label("l_optimize", 875, 200, 60, "lower"),
            label("l_codegen", 1165, 200, 60, "emit"),
            label("l_regions", 480, 420, 100, "regions", align="left"),
            label("l_score", 585, 565, 60, "score"),
            label("l_assign", 1010, 420, 80, "assign"),
            label("l_key", 1350, 420, 40, "key", align="left"),
            label("n_cost", 645, 665, 230, "cycle cost model"),
            label(
                "note",
                65,
                760,
                800,
                "The native path covers the fused MLP; other graphs run eager.",
            ),
            label("source", 65, 795, 800, "Source: frontend/python/forge_torch, version 0.2.0."),
        ]
    )

    elements.insert(0, title("How Sparrow turns a PyTorch graph into backend artifacts"))
    return scene(elements)


def runtime() -> dict:
    boxes = [
        Box("plan", 40, 260, 190, 100, "ExecutionPlan", "runtime"),
        Box("liveness", 292, 260, 190, 100, "Liveness", "runtime"),
        Box("planner", 544, 260, 190, 100, "Memory planner", "runtime"),
        Box("scheduler", 796, 260, 190, 100, "Scheduler", "runtime"),
        Box("provider", 1048, 260, 190, 100, "DeviceProvider", "runtime"),
        Box("events", 1300, 260, 190, 100, "Events", "neutral"),
    ]

    elements: list[dict] = []
    for b in boxes:
        shape, text = box(b)
        elements.extend([shape, text])

    elements.extend(
        [
            arrow("a_analyze", 230, 310, 292, 310),
            arrow("a_ranges", 482, 310, 544, 310),
            arrow("a_allocate", 734, 310, 796, 310),
            arrow("a_dispatch", 986, 310, 1048, 310),
            arrow("a_complete", 1238, 310, 1300, 310),
        ]
    )

    elements.extend(
        [
            label("l_analyze", 230, 220, 62, "analyze"),
            label("l_ranges", 463, 220, 100, "live ranges"),
            label("l_allocate", 715, 220, 100, "allocations"),
            label("l_dispatch", 986, 220, 62, "dispatch"),
            label("l_complete", 1238, 220, 72, "complete"),
            label("n_planner", 544, 390, 190, "first-fit decreasing"),
            label("n_scheduler", 796, 390, 190, "Kahn topological order"),
            label("n_providers", 1048, 390, 190, "Triton, Astra,\nMicroGPU"),
            label(
                "note",
                40,
                480,
                900,
                "The frontend builds the fused MLP plan; tests build the rest.",
            ),
            label("source", 40, 515, 900, "Source: runtime/include/forge_rt, version 0.2.0."),
        ]
    )

    elements.insert(0, title("ForgeRT schedules one execution plan across providers"))
    return scene(elements)


def write(name: str, data: dict) -> None:
    path = HERE / f"{name}.excalidraw"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    write("compilation", compilation())
    write("runtime", runtime())
