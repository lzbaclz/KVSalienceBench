"""Protocol/adversarial tests: fail closed instead of manufacturing comparison evidence."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pytest

from analyze_physical_kv import analyze
from prepare_physical_qa import render_prompt, truncate_middle, token_count
from scripts.check_camera_ready import inspect_log


def cell(n_datasets=2):
    # All keys used by the strict analyzer are explicitly present.
    cfg=dict(block_size=32,sink_blocks=1,recent_blocks=1,ema_decay=.9,cross_fraction=.1,
             prefill_chunk=256,observation_window=64,budget_fraction=.2)
    return dict(status='passed',gates={'passed':True},data_sha256='data',
        checkpoint_files_sha256={'model.safetensors':'weights'},source_sha256={'core.py':'source'},
        weights_hashed=True,quality_metric='QA',predictor_sha256=None,
        args=dict(dtype='bfloat16',device='cuda:0',max_new_tokens=128,chat=True,fixed_output=False,
                  seed=0,warmup=1,policy='h2o_block'),config=cfg,
        runtime=dict(torch='2.6',transformers='4.51.3',cuda='12.4',gpu_name='A100',cpu_threads=1),
        results=[dict(id=f'{i}',dataset=f'd{i}',input_token_sha256=f'tokens{i}',f1=.5,
            output_tokens=10,local_ttft_ms=10.,local_tpot_ms=1.,local_generation_ms=19.,
            final_cache={'kv_storage_bytes':1000}) for i in range(n_datasets)])


def test_paired_small_cohort_is_descriptive():
    a=cell(); b=deepcopy(a); b['results'][0]['f1']+=.001
    out=analyze(a,b)
    assert out['paired_requests']==2 and out['quality']['equivalent'] is None
    assert out['baseline']['local_tpot_ms']['p99.9'] is None


@pytest.mark.parametrize('fault',['ids','duplicate','weights','failed','gates','source','runtime','budget','tokens','missing_f1','nan','args'])
def test_paired_comparison_rejects_invalid_evidence(fault):
    a=cell(); b=deepcopy(a)
    if fault=='ids': b['results'][0]['id']='other'
    elif fault=='duplicate': b['results'].append(b['results'][0])
    elif fault=='weights': b['weights_hashed']=False
    elif fault=='failed': b['status']='failed'
    elif fault=='gates': b['gates']['passed']=False
    elif fault=='source': b['source_sha256']={'core.py':'other'}
    elif fault=='runtime': b['runtime']['torch']='9'
    elif fault=='budget': b['config']['budget_fraction']=.3
    elif fault=='tokens': b['results'][0]['input_token_sha256']='changed'
    elif fault=='missing_f1': b['results'][0]['f1']=None
    elif fault=='nan': b['results'][0]['f1']=float('nan')
    elif fault=='args': b['args']['fixed_output']=True
    with pytest.raises(ValueError): analyze(a,b)


def test_exploratory_cluster_tost_and_zero_variance():
    a=cell(8); b=deepcopy(a)
    for i,r in enumerate(b['results']): r['f1']+= (i-3.5)*.0001
    out=analyze(a,b)
    assert out['quality']['equivalent'] is True
    assert out['quality']['mean_delta_ci90'][0]<0<out['quality']['mean_delta_ci90'][1]
    assert analyze(a,a)['quality']['equivalent'] is None


def test_fixed_output_has_no_quality_inference():
    a=cell(8)
    a['quality_metric']=None; a['args']['fixed_output']=True
    for r in a['results']: r['f1']=None
    assert analyze(a,a)['quality']['number_of_dataset_clusters']==0


class TinyTokenizer:
    def __call__(self,text,add_special_tokens=True):
        class Enc: pass
        x=Enc(); x.input_ids=[ord(c) for c in text]+([1] if add_special_tokens else [])
        return x
    def decode(self,ids,skip_special_tokens=True): return ''.join(chr(i) for i in ids)
    def apply_chat_template(self,messages,**kwargs):
        return [1,2]+[ord(c) for c in messages[0]['content']]+[3,4]


def test_explicit_truncation_accounts_for_chat_overhead():
    t=TinyTokenizer(); s='abcdefghijklmnopqrstuvwxyz'
    clipped=truncate_middle(t,s,12,True)
    assert token_count(t,clipped,True)<=12
    assert clipped.startswith('a') and clipped.endswith('z')
    with pytest.raises(ValueError): truncate_middle(t,s,1,True)
    assert render_prompt({'prompt':'ready'})=='ready'
    with pytest.raises(ValueError): render_prompt({'context':'missing question'})


def test_preparer_offline_cli(tmp_path,monkeypatch):
    transformers=pytest.importorskip('transformers')
    from prepare_physical_qa import main
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:TinyTokenizer())
    source=tmp_path/'source.jsonl'
    rows=[dict(id=str(i),prompt='context '*10,answers=['yes']) for i in range(5)]
    source.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    out=tmp_path/'frozen.jsonl'
    args=['--model','/unused','--input',f'qa={source}','--out',str(out),
          '--per-dataset','3','--max-input-tokens','20','--chat','--truncate-middle']
    assert main(args)==0
    meta=json.loads(Path(str(out)+'.meta.json').read_text())
    assert len(meta['requests'])==3 and all(r['truncated'] for r in meta['requests'])
    with pytest.raises(FileExistsError): main(args)


def test_pdf_log_guard():
    assert inspect_log('Output written on main.pdf (7 pages).')==[]
    assert inspect_log(r'Overfull \hbox (1pt too wide)')
    assert inspect_log("LaTeX Warning: Reference `x' on page 1 undefined")
    assert inspect_log('! LaTeX Error: bad')


def test_query_analysis_requires_v2_and_holds_out_requests(tmp_path):
    from run_query_control_v2 import analyze_trace
    from xqp.physical_validation import sha256_file
    trace=tmp_path/'trace.jsonl'
    rng=np.random.default_rng(2)
    with trace.open('w') as f:
        for request in range(16):
            for i in range(20):
                y=i%2; within=float(y+rng.normal(0,.7))
                f.write(json.dumps(dict(trace_version=2,request_id=f'p{request}',f_within=within,
                    f_cross=float(rng.normal()),f_query_dotmax=float(rng.normal()),y_h4=y))+'\n')
    meta=Path(str(trace)+'.meta.json')
    meta.write_text(json.dumps(dict(trace_version=2,trace_sha256=sha256_file(trace),requests=list(range(16)))))
    out=analyze_trace(trace,bootstraps=100)
    assert set(out['train_ids']).isdisjoint(out['test_ids'])
    assert len(out['test_ids'])==4 and np.isfinite(out['mean_request_auc_delta'])
    meta.write_text(json.dumps(dict(trace_version=1,trace_sha256=sha256_file(trace))))
    with pytest.raises(ValueError,match='version-2'): analyze_trace(trace)


def test_committed_physical_kv_summaries_match_their_raw_cells():
    """The manuscript's Llama/Qwen tables are read off `summary.json`; that file
    must stay a pure function of the tracked per-request cells."""
    from summarize_physical_kv import main
    root = Path(__file__).resolve().parents[1] / 'experiments/results/physical_kv'
    runs = [('llama3.1-v1', 'llama', 'Llama-3.1-8B-Instruct'),
            ('qwen25-v1', 'qwen', 'Qwen2.5-7B-Instruct')]
    for run, tag, model in runs:
        note = json.loads((root / run / 'summary.json').read_text()).get('note', '')
        assert main(['--run-dir', str(root / run),
                     '--qa-meta', str(root / f'physical-qa-{tag}-v1.jsonl.meta.json'),
                     '--query-analysis', str(root / f'query-v2/query-v2.{tag}.bs32.analysis.json'),
                     '--query-meta', str(root / f'query-v2/query-v2.{tag}.bs32.jsonl.meta.json'),
                     '--model', model, '--note', note, '--check']) == 0


def test_summary_generator_rejects_a_gate_only_or_mismatched_cohort(tmp_path):
    from summarize_physical_kv import build
    root = Path(__file__).resolve().parents[1] / 'experiments/results/physical_kv'
    run = tmp_path / 'run'
    run.mkdir()
    source = json.loads((root / 'qwen25-v1/full.r0.json').read_text())
    gate_only = deepcopy(source); gate_only['results'] = []
    (run / 'full.r0.json').write_text(json.dumps(gate_only))
    with pytest.raises(ValueError, match='gate-only'):
        build(run, '', root / 'physical-qa-qwen-v1.jsonl.meta.json', None, None, None)
    (run / 'full.r0.json').write_text(json.dumps(source))
    with pytest.raises(ValueError, match='prompt cohort'):
        build(run, '', root / 'physical-qa-llama-v1.jsonl.meta.json', None, None, None)


def test_oracle_budget_rejects_unusable_uniform_count_interpretation(tmp_path):
    """A miss below the logged-count diagnostic requires an aggregation audit."""
    from analyze_oracle_budget import per_request, summarize
    cell = tmp_path / "ds_h2o.json"
    good = dict(id="0", per_step_block_count=[26, 26], per_step_B_t_oracle=[32, 32],
                per_step_eps_measured=[0.5, 0.6], budget_frac=0.2)
    cell.write_text(json.dumps(dict(context_length=4096, results=[good])))
    rows = per_request(cell)
    assert rows[0]["forced_floor_mean"] == pytest.approx(1 - 26 / 32)
    assert summarize(rows)["n_requests"] == 1
    bad = dict(good, per_step_eps_measured=[0.0, 0.0])
    cell.write_text(json.dumps(dict(context_length=4096, results=[bad])))
    with pytest.raises(ValueError, match="below logged-count diagnostic"):
        per_request(cell)
    ragged = dict(good, per_step_B_t_oracle=[32])
    cell.write_text(json.dumps(dict(context_length=4096, results=[ragged])))
    with pytest.raises(ValueError, match="ragged"):
        per_request(cell)


def test_decision_operating_point_bootstrap_is_paired():
    from analyze_decision_operating_point import paired_bootstrap
    diff = np.full(64, 0.05)
    out = paired_bootstrap(diff, 200, 0)
    assert out["mean"] == pytest.approx(0.05)
    assert out["n_positive"] == 64 and out["n_negative"] == 0
    assert out["ci95"][0] == pytest.approx(0.05) and out["ci95"][1] == pytest.approx(0.05)
    mixed = np.array([0.1] * 40 + [-0.1] * 24)
    out = paired_bootstrap(mixed, 500, 0)
    assert out["n_positive"] == 40 and out["n_negative"] == 24
    assert out["ci95"][0] < out["mean"] < out["ci95"][1]
