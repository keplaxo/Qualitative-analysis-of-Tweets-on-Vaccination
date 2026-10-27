"""No real tweets: validate full-corpus hash routing and GPU capacity gates."""
import ast,hashlib,importlib.util
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def load(path,module):
    spec=importlib.util.spec_from_file_location(module,ROOT/path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_index_date_mapping_and_source_schema():
    idx=load("kaggle/full_corpus_index.py","corpus_index_contract")
    assert idx.column(["Tweet","created_at"],idx.TEXT_NAMES)=="Tweet"
    assert idx.column(["Tweet","created_at"],idx.DATE_NAMES)=="created_at"
    assert idx.column(["x","y"],idx.DATE_NAMES) is None
    assert idx.sanitized_years(pd.Series([
        "2010-01-02","2019-12-31T23:59:59Z","", "bad", "1995-01-01"]
    ))==[2010,2019,0,0,0]

def test_hash_partition_coverage_and_determinism():
    texts=["Vaccination now","Vaccine trial","No public statement",
           "Vaccination now"]
    keys=[hashlib.sha256(t.encode("utf-8")).digest() for t in texts]
    assert keys[0]==keys[-1]
    shards=[digest[0]>>2 for digest in keys]
    assert 0<=min(shards)<=max(shards)<64
    assert shards[0]==shards[-1]
    assert len(set(keys))==3

def test_controller_gates_and_kernel_names():
    v=load("scripts/advance_full_corpus.py","corpus_controller_contract")
    assert v.make_ref("sentiment",0)==(
        "keplaxo/vax-corpus-sentiment-00-20261019")
    assert v.make_ref("stance",63)==(
        "keplaxo/vax-corpus-stance-63-20261019")
    fake={"input_unique_n":1000,"english_screened_n":700,
          "runtime_seconds":{"xlmt_seconds":25,
                             "roberta_seconds":22,"deberta_seconds":250}}
    assert v.estimate("sentiment",2000,fake)==788.0
    assert v.estimate("stance",2000,fake)==1600.0
    assert v.MAX_SECONDS<18000
    assert v.COUNT==64

def test_gpu_script_injection_and_import_syntax():
    source=(ROOT/"kaggle/full_corpus_gpu_stage.py").read_text()
    assert source.count('MODE="MODE_PLACEHOLDER"')==1
    assert source.count('SHARD_INDEX=SHARD_PLACEHOLDER')==1
    for mode in ("pilot","sentiment","stance"):
        compiled=source.replace('MODE="MODE_PLACEHOLDER"',
                                'MODE="'+mode+'"').replace(
                                    'SHARD_INDEX=SHARD_PLACEHOLDER',
                                    "SHARD_INDEX=0")
        ast.parse(compiled)
        assert "accuracy_measured" in compiled
        assert "stance_validated" in compiled


def test_cpu_reporting_scripts_parse_and_wave_injection():
    import ast
    wave=(ROOT/"kaggle/full_corpus_aggregate_wave.py").read_text()
    final=(ROOT/"kaggle/full_corpus_aggregate_final.py").read_text()
    assert wave.count("WAVE_INDEX=WAVE_PLACEHOLDER")==1
    for i in (0,7,15):
        ast.parse(wave.replace("WAVE_INDEX=WAVE_PLACEHOLDER",
                               "WAVE_INDEX="+str(i)))
    ast.parse(final)
    assert "accuracy_measured" in final
    assert "stance_prevalence_validated" in final

def test_final_controller_wave_stage_identifiers_and_size():
    controller=load("scripts/advance_full_corpus.py","corpus_aggregate_contract")
    assert controller.wave_ref(0)=="keplaxo/vax-corpus-report-wave-00-20261019"
    assert controller.wave_ref(15)=="keplaxo/vax-corpus-report-wave-15-20261019"
    assert controller.WAVES==16
    assert controller.COUNT==64

def test_nested_kaggle_summary_download_pattern(monkeypatch,tmp_path):
    """Kaggle output names are nested; the CLI requires a regex rather than a basename."""
    import json,re,types
    controller=load("scripts/advance_full_corpus.py","corpus_regex_recovery")
    calls=[]
    def command(*args,timeout=110):
        assert args[:2]==("kernels","output")
        pattern=args[args.index("--file-pattern")+1]
        assert re.search(pattern,"vax_outputs/corpus_shard_report.json")
        target=Path(args[args.index("-p")+1])/"vax_outputs"/"corpus_shard_report.json"
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps({"mode":"sentiment","shard":14}))
        calls.append(pattern)
        return types.SimpleNamespace(returncode=0,stdout="",stderr="")
    monkeypatch.setattr(controller,"command",command)
    assert controller.read_summary(
        "keplaxo/vax-corpus-sentiment-14-20261019",
        "corpus_shard_report.json")=={"mode":"sentiment","shard":14}
    assert len(calls)==1


def test_summary_filename_allowlist():
    controller=load("scripts/advance_full_corpus.py","corpus_report_allowlist")
    import pytest
    with pytest.raises(RuntimeError,match="Unapproved"):
        controller.read_summary("private/record","raw_tweet_data.csv")

def test_transient_kaggle_summary_retry_never_launches_kernel(monkeypatch):
    """A temporary download error can recover without any inference resubmission."""
    import json,types
    controller=load("scripts/advance_full_corpus.py","corpus_retry_recovery")
    trials=[]
    def command(*args,timeout=110):
        assert args[:2]==("kernels","output")
        trials.append(args)
        if len(trials)==1:
            return types.SimpleNamespace(returncode=1,stdout="",stderr="simulated transient")
        target=Path(args[args.index("-p")+1])/"vax_outputs"/"corpus_shard_report.json"
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps({"coverage_complete":True}))
        return types.SimpleNamespace(returncode=0,stdout="",stderr="")
    monkeypatch.setattr(controller,"command",command)
    monkeypatch.setattr(controller.time,"sleep",lambda _:None)
    assert controller.read_summary(
        "keplaxo/vax-corpus-sentiment-11-20261019",
        "corpus_shard_report.json")=={"coverage_complete":True}
    assert len(trials)==2
