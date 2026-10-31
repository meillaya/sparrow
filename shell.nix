{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/61b7c44c4073f0b827768aff0049561b5110ea5a.tar.gz") {} }:
pkgs.mkShell {
  packages = with pkgs; [
    cmake ninja gnumake pkg-config
    gtest nlohmann_json
    python312
  ] ++ (with pkgs.python312Packages; [
    torch
    pytest
    lit
    numpy
    pybind11
  ]);
  shellHook = ''
    echo "Sparrow Phase 1 dev environment activated (shell.nix fallback)"
  '';
}
