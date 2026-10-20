"""Bridge to Part 1's code.

Both parts use a package called `src`, so Part 1's package is loaded under the alias `p1`
(its modules use relative imports, so they work under any package name). After
`import src.p1`, Part 1 is importable as `from p1.split import get_split`, etc.
Part 1's own paths (data/, splits/) still resolve to part1/, so splits are shared.
"""
import importlib.util
import sys
from pathlib import Path

PART2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PART2_ROOT.parent
PART1_ROOT = REPO_ROOT / "part1"


def _load_part1():
    if "p1" in sys.modules:
        return sys.modules["p1"]
    spec = importlib.util.spec_from_file_location(
        "p1", PART1_ROOT / "src" / "__init__.py", submodule_search_locations=[str(PART1_ROOT / "src")])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p1"] = mod
    spec.loader.exec_module(mod)
    return mod


_load_part1()
