{ pkgs, ... }:

{
  cachix.enable = false;

  env.NIXPKGS_ALLOW_UNFREE = "1";

  # Phase 1 dev toolchain (2026-07-18): uv + ruff + ty were added on top of the
  # original pip+venv surface. The Python deps themselves stay on PyPI and are
  # installed into .venv via `uv pip install` (see scripts/setup-env.sh); the
  # Nix copy in python312Packages.{torch,xdsl,...} is only used when this shell
  # is the active env. ruff and ty are also installed system-wide via
  # `uv tool install` so they show up on ~/.local/bin regardless of whether
  # the venv is activated.
  packages = [
    # Build / lint / type-check toolchain
    pkgs.uv
    pkgs.ruff
    pkgs.ty

    # Native build deps (cmake/ninja) + system test headers
    pkgs.cmake
    pkgs.ninja
    pkgs.gnumake
    pkgs.pkg-config

    pkgs.python312
    pkgs.python312Packages.pytorch
    pkgs.python312Packages.xdsl
    pkgs.python312Packages.pytest
    pkgs.python312Packages.lit
    pkgs.python312Packages.numpy
    pkgs.python312Packages.pybind11

    pkgs.gtest
    pkgs.nlohmann_json
  ];

  enterShell = ''
    echo "ForgeStack Phase 1 dev environment activated"
    echo "  python: $(python3 --version 2>&1)"
    echo "  uv:     $(uv --version 2>&1)"
    echo "  ruff:   $(ruff --version 2>&1)"
    echo "  ty:     $(ty --version 2>&1)"
  '';
}