# compiler/lib/Dialect — xDSL dialect implementations.
#
# Each dialect lives in its own sub-package (e.g. `Dialect.Tensor`,
# `Dialect.Graph`); each `__init__.py` exports a `Dialect` object that
# `tools/forge-opt/forge_opt.py` auto-discovers via importlib.
