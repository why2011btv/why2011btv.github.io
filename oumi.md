### to login
```ssh perlmutter```

### to reserve a node
```salloc --nodes 1 --qos interactive --time 00:15:00 --constraint gpu --gpus 4```

### to activate env
```
cd $SCRATCH/nanochat
source .venv/bin/activate
```
```
(nanochat) (oumi) why16gzl@nid001005:/pscratch/sd/w/why16gzl/nanochat> which python
/pscratch/sd/w/why16gzl/nanochat/.venv/bin/python
```

#### On Dec 12 2025, I downloaded 240 parquet to this path: /pscratch/sd/w/why16gzl/.cache/nanochat/base_data

```python -m nanochat.dataset -n 240```

#### And I also trained the tokenizer:
```
(nanochat) (oumi) why16gzl@nid001005:/pscratch/sd/w/why16gzl/nanochat> python -m scripts.tok_train --max_chars=2000000000

max_chars: 2,000,000,000
doc_cap: 10,000
vocab_size: 65,536
2025-12-26 09:53:41,571 - rustbpe - INFO - Processing sequences from iterator (buffer_size: 8192)
2025-12-26 09:53:41,576 - rustbpe - INFO - Processed 0 sequences total, 0 unique
2025-12-26 09:53:41,576 - rustbpe - INFO - Starting BPE training: 65271 merges to compute
2025-12-26 09:53:41,576 - rustbpe - INFO - Computing initial pair counts from 0 unique sequences
2025-12-26 09:53:41,582 - rustbpe - INFO - Building heap with 0 unique pairs
2025-12-26 09:53:41,582 - rustbpe - INFO - Starting merge loop
2025-12-26 09:53:41,582 - rustbpe - INFO - Finished training: 0 merges completed
Training time: 0.03s
Saved tokenizer encoding to /global/homes/w/why16gzl/.cache/nanochat/tokenizer/tokenizer.pkl
Saved token_bytes to /global/homes/w/why16gzl/.cache/nanochat/tokenizer/token_bytes.pt
```


```
(nanochat) (oumi) why16gzl@nid001005:/pscratch/sd/w/why16gzl/.cache/nanochat/tokenizer> salloc: Job 46975710 has exceeded its time limit and its allocation has been revoked.
                                                                                                                                                                             slurmstepd: error: *** STEP 46975710.interactive ON nid001005 CANCELLED AT 2025-12-26T17:55:54 DUE TO TIME LIMIT ***
srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
exit
(nanochat) (oumi) why16gzl@perlmutter:login05:/pscratch/sd/w/why16gzl/nanochat> python -m scripts.tok_eval

Traceback (most recent call last):
  File "/global/homes/w/why16gzl/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/global/homes/w/why16gzl/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/pscratch/sd/w/why16gzl/nanochat/scripts/tok_eval.py", line 6, in <module>
    from nanochat.dataset import parquets_iter_batched
  File "/pscratch/sd/w/why16gzl/nanochat/nanochat/dataset.py", line 17, in <module>
    from nanochat.common import get_base_dir
  File "/pscratch/sd/w/why16gzl/nanochat/nanochat/common.py", line 9, in <module>
    import torch
  File "/pscratch/sd/w/why16gzl/nanochat/.venv/lib/python3.10/site-packages/torch/__init__.py", line 427, in <module>
    from torch._C import *  # noqa: F403
ImportError: /pscratch/sd/w/why16gzl/nanochat/.venv/lib/python3.10/site-packages/torch/lib/libc10_cuda.so: undefined symbol: cudaGetDriverEntryPointByVersion, version libcudart.so.12

```
