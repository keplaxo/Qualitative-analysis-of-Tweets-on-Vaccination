"""Offline invariants for the staged, private Kaggle Qwen research workflow."""
from pathlib import Path
import ast
import json

ROOT=Path(__file__).resolve().parents[1]

def test_source_scripts_compile_without_kaggle_runtime():
    for rel in ("scripts/advance_kaggle.py","kaggle/qwen_stage.py",
                "kaggle/report_stage.py"):
        ast.parse((ROOT/rel).read_text(encoding="utf-8"),filename=rel)

def test_stage_modes_and_controls():
    tree=ast.parse((ROOT/"scripts/advance_kaggle.py").read_text())
    assignments={node.targets[0].id:node.value for node in tree.body
                 if isinstance(node,ast.Assign)
                 and isinstance(node.targets[0],ast.Name)}
    stages=ast.literal_eval(assignments["STAGES"])
    assert [s["name"] for s in stages]==["pilot","full","report"]
    assert [s["gpu"] for s in stages]==[True,True,False]
    assert "keplaxo/vax-sentiment-benchmark-full" in stages[0]["sources"]
    assert "keplaxo/vax-qwen-pilot" in stages[1]["sources"]
    assert "keplaxo/vax-qwen-300" in stages[2]["sources"]
    assert not any("master.csv" in src for stage in stages for src in stage["sources"])

def test_private_checkpoint_and_quality_guards_present():
    runner=(ROOT/"scripts/advance_kaggle.py").read_text(encoding="utf-8")
    semantic=(ROOT/"kaggle/qwen_stage.py").read_text(encoding="utf-8")
    report=(ROOT/"kaggle/report_stage.py").read_text(encoding="utf-8")
    assert '"is_private":True' in runner
    assert "if stage[\"kernel\"].lower() in existing()" in runner
    assert "schema_valid_count" in runner
    assert "enabled" in runner
    assert "max-records" in semantic
    assert "QWEN_EXPECTED=20 if MODE" in semantic
    assert "shutil.copyfile(pilot_checkpoint,checkpoint)" in semantic
    assert "schema_valid_fraction" in semantic
    assert "accuracy_measured" in report
    assert "generalizable_to_corpus" in report
