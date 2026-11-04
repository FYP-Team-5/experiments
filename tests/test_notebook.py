import ast
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_run_comparison_notebook_cells_parse():
    with open(os.path.join(ROOT, "run_comparison.ipynb"), encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for source in code_cells:
        ast.parse(source)
    assert any("CONFIGS = [" in source for source in code_cells)
