#!/usr/bin/env python3
"""forge-opt — ForgeStack's xDSL-based MLIR driver.

Phase 1 (Task 5) supports:

* ``--action=roundtrip <file.mlir>`` — parse the file and print it back to
  stdout, after running the verifier. Used as the canonical "the dialect
  parses + prints + verifies" smoke test.

Future actions (pass pipelines, conversion to ``forge.tensor``, …) land in
later tasks; the CLI shape is forward-compatible.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
from pathlib import Path
from types import ModuleType

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func
from xdsl.ir import Dialect
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import DiagnosticException, ParseError, VerifyException

HERE = Path(__file__).resolve().parent
LIB_ROOT = HERE.parent.parent / "lib"
DIALECT_ROOT = LIB_ROOT / "Dialect"

# ``compiler/tools/forge-opt/`` has a dash, so it can't be a Python
# package itself; add the directory that *contains* ``compiler/`` and
# import ``compiler.lib.Dialect.<X>`` as a regular sub-package.
_PARENT = LIB_ROOT.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))


def _load_dialect_module(name: str) -> ModuleType | None:
    """Import a single ``compiler.lib.Dialect.<name>`` sub-package."""
    fq_name = f"compiler.lib.Dialect.{name}"
    try:
        return importlib.import_module(fq_name)
    except ModuleNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover — surfaced to the user
        print(f"error: failed to import {fq_name}: {exc}", file=sys.stderr)
        return None


def _extract_dialect(mod: ModuleType, package_name: str) -> Dialect | None:
    """Pull a ``Dialect`` instance out of a dialect module.

    Accepts three conventions, in priority order:

    * ``mod.<PackageName>`` (xDSL short name, e.g. ``mod.Tensor``)
    * ``mod.<PackageName>Dialect``
    * ``mod.Dialect``

    The lookup also rejects bare classes (e.g. someone re-exported
    ``xdsl.ir.Dialect`` without ever instantiating it) so that a half-baked
    module is silently skipped rather than crashing the driver.
    """
    candidates: tuple[str, ...] = (
        package_name,
        f"{package_name}Dialect",
        "Dialect",
    )
    for attr in candidates:
        candidate = getattr(mod, attr, None)
        if candidate is None or isinstance(candidate, (property, type)):
            continue
        if getattr(candidate, "operations", None) is None:
            continue
        return candidate
    return None


def discover_dialects() -> list[tuple[str, Dialect]]:
    """Walk ``compiler/lib/Dialect/`` and import every sub-package.

    Returns a list of ``(dialect_name, Dialect_obj)`` tuples; sub-packages
    without a recognisable ``Dialect`` attribute are skipped silently.
    """
    found: list[tuple[str, Dialect]] = []
    if not DIALECT_ROOT.is_dir():
        return found
    for entry in sorted(DIALECT_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "__init__.py").exists():
            continue
        mod = _load_dialect_module(entry.name)
        if mod is None:
            continue
        dialect = _extract_dialect(mod, entry.name)
        if dialect is None:
            continue
        found.append((dialect.name, dialect))
    return found


def build_context(
    *,
    auto_dialects: bool = True,
    all_dialects: bool = False,
    extra_dialects: list[str] | None = None,
) -> Context:
    """Build an xDSL ``Context`` with the dialects Phase 1 needs."""
    ctx = Context(allow_unregistered=False)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    if auto_dialects:
        for _name, dialect in discover_dialects():
            ctx.load_dialect(dialect)
    if all_dialects:
        from xdsl.universe import get_all_dialects

        for dialect_obj in get_all_dialects():
            if not isinstance(dialect_obj, Dialect):
                continue
            with contextlib.suppress(Exception):
                ctx.load_dialect(dialect_obj)
    for name in extra_dialects or ():
        mod = importlib.import_module(name)
        # Mirror `_extract_dialect` — accept the short name (`Tensor`), the
        # `<X>Dialect` alias, or a bare `Dialect` instance on the module.
        package_name = name.rsplit(".", 1)[-1]
        candidate = _extract_dialect(mod, package_name)
        if candidate is None:
            print(f"error: no Dialect found in {name!r}", file=sys.stderr)
            continue
        ctx.load_dialect(candidate)
    return ctx


def run_roundtrip(ctx: Context, source: str, source_name: str) -> ModuleOp:
    """Parse, verify, and re-print the module."""
    parser = Parser(ctx, source, source_name)
    module = parser.parse_module()
    module.verify()
    Printer(stream=sys.stdout).print_op(module)
    return module


ACTIONS = {
    "roundtrip": run_roundtrip,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forge-opt",
        description="ForgeStack xDSL dialect driver (Phase 1).",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Input MLIR file ('-' for stdin).",
    )
    parser.add_argument(
        "--action",
        choices=sorted(ACTIONS),
        default="roundtrip",
        help="Action to perform on the parsed module (default: roundtrip).",
    )
    parser.add_argument(
        "--pass-pipeline",
        default=None,
        help="Reserved for future use (Task 7 will wire graph-to-tensor).",
    )
    parser.add_argument(
        "--no-auto-dialects",
        action="store_true",
        help="Skip auto-discovery of compiler/lib/Dialect/*.",
    )
    parser.add_argument(
        "--all-dialects",
        action="store_true",
        help="Also load every dialect xDSL ships in `xdsl.universe`.",
    )
    parser.add_argument(
        "--load-dialect",
        action="append",
        default=[],
        metavar="FQN",
        help="Fully-qualified dialect module to import and load "
        "(repeatable, e.g. `compiler.lib.Dialect.Tensor`).",
    )
    parser.add_argument(
        "--split-input-file",
        action="store_true",
        help="Process chunks separated by `// -----` independently. "
        "Useful when a single test file contains multiple negative cases "
        "and we want each one to fail in isolation rather than have the "
        "first parse error mask the others.",
    )
    args = parser.parse_args(argv)

    if args.input == "-":
        source = sys.stdin.read()
        source_name = "<stdin>"
    else:
        source = Path(args.input).read_text()
        source_name = args.input

    ctx = build_context(
        auto_dialects=not args.no_auto_dialects,
        all_dialects=args.all_dialects,
        extra_dialects=args.load_dialect,
    )

    try:
        if args.split_input_file:
            _run_split_input(ctx, source, source_name, ACTIONS[args.action])
            return 0
        ACTIONS[args.action](ctx, source, source_name)
    except ParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except VerifyException as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except DiagnosticException as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    return 0


def _run_split_input(ctx: Context, source: str, source_name: str, action) -> None:
    """
    Run `action` on each ``// -----``-separated chunk of `source`.

    Each chunk runs independently — a failure on one chunk does NOT mask
    failures on later chunks, since we want every negative test in the
    same file to surface its own FileCheck pattern. Returns the highest
    exit code observed; the caller is expected to convert that to a
    process exit code.
    """
    chunks = source.split("// -----")
    worst = 0
    for i, chunk in enumerate(chunks):
        if i > 0:
            print("// -----")
        try:
            action(ctx, chunk, source_name)
        except ParseError as exc:
            print(f"error: {exc}", file=sys.stderr)
            worst = max(worst, 1)
        except VerifyException as exc:
            print(f"error: {exc}", file=sys.stderr)
            worst = max(worst, 2)
        except DiagnosticException as exc:
            print(f"error: {exc}", file=sys.stderr)
            worst = max(worst, 3)
    if worst:
        raise SystemExit(worst)


if __name__ == "__main__":
    raise SystemExit(main())
