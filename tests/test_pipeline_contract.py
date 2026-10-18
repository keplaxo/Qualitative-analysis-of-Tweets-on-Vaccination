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
    runner=(ROOT/"scripts/advance_kaggle.py").read_text(encoding="utf-8")
    # Stage order and dependencies can be checked without executing an API call.
    assert runner.index('"name":"pilot"') < runner.index('"name":"full"')
    assert runner.index('"name":"full"') < runner.index('"name":"report"')
    assert '"kernel":"keplaxo/vax-qwen-pilot"' in runner
    assert '"kernel":"keplaxo/vax-qwen-300"' in runner
    assert '"kernel":"keplaxo/vax-semantic-research-report"' in runner
    assert '"gpu":False' in runner
    assert 'SRC="keplaxo/vax-sentiment-benchmark-full"' in runner
    assert '"sources":[SRC,"keplaxo/vax-qwen-pilot"]' in runner
    assert '"sources":[SRC,"keplaxo/vax-qwen-300"]' in runner

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
