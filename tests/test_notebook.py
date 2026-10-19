"""Check that the published Kaggle notebook contains parseable Python cells."""
import ast
import json
from pathlib import Path

def test_notebook_structure_and_code():
    notebook = Path(__file__).resolve().parents[1]/"kaggle"/"vaccination_benchmark.ipynb"
    doc = json.loads(notebook.read_text(encoding="utf-8"))
    assert doc["nbformat"] == 4
    assert len(doc["cells"]) >= 8
    sources = []
    for cell in doc["cells"]:
        code = "".join(cell["source"])
        if cell["cell_type"] == "code":
            ast.parse(code)
            sources.append(code)
    joined = "\n".join(sources)
    assert "vaxbench.triage" in joined
    assert "vaxbench.summarize" in joined
    assert "RUN_QWEN" in joined
    assert "MASTER_CSV" in joined
