import ast
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_run_comparison_notebook_cells_parse():
    with open(os.path.join(ROOT, "notebooks", "run_comparison.ipynb"), encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for source in code_cells:
        ast.parse(source)
    assert any("CONFIGS = [" in source for source in code_cells)


def test_judge_notebook_cells_parse():
    with open(os.path.join(ROOT, "notebooks", "judge_feedback_27b.ipynb"), encoding="utf-8") as handle:
        notebook = json.load(handle)
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 4
    for source in code_cells:
        ast.parse(source)
    assert any("judge.build_items" in source for source in code_cells)
