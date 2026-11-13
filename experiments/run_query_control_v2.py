#!/usr/bin/env python3
"""Collect/inspect phase-aligned query probes. NOT the Quest page-bound algorithm.

New v2 output only, no old .done files, no silent feature imputation. The
exploratory estimand is mean HELD-OUT REQUEST AUC, not pooled block AUC.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from xqp.physical_validation import read_requests, sha256_file


def analyze_trace(path, seed=0, bootstraps=2000):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    path=Path(path)
    meta=json.loads(Path(str(path)+'.meta.json').read_text())
    if meta.get('trace_version') != 2 or meta.get('trace_sha256') != sha256_file(path):
        raise ValueError('require complete, checksum-matching version-2 trace; legacy traces are not accepted')
    if bootstraps < 1: raise ValueError('positive bootstrap count required')
    fields=('f_within','f_cross','f_query_dotmax')
    groups={}
    with path.open() as f:
        for line in f:
            row=json.loads(line)
            if row.get('trace_version') != 2: raise ValueError('mixed trace versions')
            values=[float(row[k]) for k in fields]
            if not np.isfinite(values).all() or row['y_h4'] not in (0,1):
                raise ValueError('invalid feature or label')
            groups.setdefault(row['request_id'], []).append((*values, row['y_h4']))
    ids=sorted(groups)
    if len(ids)<16: raise ValueError('need at least 16 independent request IDs (12 train / >=4 held out)')
    if len(ids)!=len(meta['requests']): raise ValueError('trace/manifest request count mismatch')
    rng=np.random.default_rng(seed)
    perm=rng.permutation(ids); n_test=max(4,len(ids)//4)
    test_ids=sorted(perm[:n_test]); train_ids=sorted(perm[n_test:])
    train=np.concatenate([np.asarray(groups[k],dtype=np.float64) for k in train_ids])
    if len(np.unique(train[:,-1]))<2: raise ValueError('training labels contain one class')
    predictors=[]
    for n in (2,3):
        predictor=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        # Treat convergence warnings as failed analysis, not successful evidence.
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            predictor.fit(train[:,:n],train[:,-1])
        predictors.append(predictor)
    rows=[]
    for rid in test_ids:
        data=np.asarray(groups[rid],dtype=np.float64)
        if len(np.unique(data[:,-1]))<2:
            raise ValueError(f'{rid}: one-class held-out request; no silent dropping')
        scores=[float(roc_auc_score(data[:,-1], p.predict_proba(data[:,:n])[:,1]))
                for p,n in zip(predictors,(2,3))]
        rows.append(dict(id=rid,two_view_auc=scores[0],plus_dotmax_auc=scores[1],delta=scores[1]-scores[0]))
    diffs=np.array([r['delta'] for r in rows])
    draws=rng.choice(diffs, size=(bootstraps,len(diffs)),replace=True).mean(1)
    return dict(protocol='phase-aligned-query-v2',trace_sha256=meta['trace_sha256'],seed=seed,
        train_ids=train_ids,test_ids=test_ids,requests=rows,mean_request_auc_delta=float(diffs.mean()),
        exploratory_request_bootstrap_ci95=np.quantile(draws,[.025,.975]).tolist(),
        estimand='mean held-out request AUC gain from adding scaled post-RoPE token/head dot-max to two views',
        limitations='Exploratory frozen-trace predictor comparison, not an information-sufficiency certificate, intrinsic query redundancy, closed-loop selection, or Quest implementation. Non-significance is not equivalence.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', required=True, help='new trace path, or completed v2 trace for --analyze-only')
    p.add_argument('--out', required=True, help='new analysis JSON')
    p.add_argument('--model'); p.add_argument('--data')
    p.add_argument('--block-size', type=int, default=32)
    p.add_argument('--feature-steps', type=int, default=32)
    p.add_argument('--max-input-tokens', type=int, default=4096)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['float32','float16','bfloat16'], default='bfloat16')
    p.add_argument('--chat', action='store_true')
    p.add_argument('--analyze-only', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args=p.parse_args(argv)
    if Path(args.out).exists(): raise FileExistsError(args.out)
    if not args.analyze_only:
        if not args.model or not args.data: p.error('--model and --data required for collection')
        import torch
        from transformers import AutoTokenizer
        from xqp.attn_trace_extract import extract_attention_traces
        requests=read_requests(args.data)
        if len(requests)<16: raise ValueError('at least 16 frozen requests required')
        tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
        ids=[]
        for row in requests:
            t=(tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,
                 add_generation_prompt=True,return_tensors='pt') if args.chat
                 else tok(row['prompt'],return_tensors='pt').input_ids)
            if t.shape[1]>args.max_input_tokens: raise ValueError('input exceeds cap; freeze explicit truncation first')
            ids.append(t)
        torch.manual_seed(args.seed)
        extract_attention_traces(args.model,out_path=args.trace,input_ids=ids,device=args.device,
            block_size=args.block_size,max_new_tokens=args.feature_steps,dtype=args.dtype,query_variants=True)
        # Additional mapping does not replace or reinterpret collector metadata.
        weights=sorted(Path(args.model).glob('*.safetensors'))+sorted(Path(args.model).glob('*.bin'))
        if not weights: raise ValueError('no local weight files for provenance')
        with Path(str(args.trace)+'.cohort.json').open('x') as f:
            json.dump(dict(data_sha256=sha256_file(args.data),args=vars(args),
                model_weight_sha256={x.name:sha256_file(x) for x in weights},
                ids=[dict(trace_id=f'p{i}',dataset=r['dataset'],id=r['id']) for i,r in enumerate(requests)]), f,indent=2)
    result=analyze_trace(args.trace,args.seed)
    cohort=Path(str(args.trace)+'.cohort.json')
    result['cohort_manifest_sha256']=sha256_file(cohort) if cohort.exists() else None
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with Path(args.out).open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    return 0


if __name__=='__main__': raise SystemExit(main())
