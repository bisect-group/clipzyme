"""Run the original scripts/main.py with seven wrapper hooks.

1. Lightning writes checkpoint_last.pt next to the other run files (resumable state).
2. EMULATOR_RESUME_CKPT, when set, resumes training from that file.
3. With EMULATOR_CHECKPOINT_ESM set, each ESM-2 layer recomputes its activations in the backward pass. The numbers stay the same and
   the activation memory drops, so batch 8 fits a 23 GB GPU.
4. With EMULATOR_FAST set, the exact speed-ups of fast_esm.py apply. Recomputation then runs only for large micro-batches,
   or for every micro-batch when EMULATOR_CHECKPOINT_ESM is also set.
5. The size of the train and dev datasets that the original code builds goes to EMULATOR_COUNTS_PATH.
6. With EMULATOR_FROZEN_ESM set to a cache folder, the frozen ESM-2 vectors come from that cache (esm_cache.py).
7. With EMULATOR_VIRTUAL_GPUS set, the CLIP loss emulates that many GPUs on one GPU (virtual_ranks.py).
"""

import atexit
import json
import os
import runpy
import sys
from pathlib import Path

import pytorch_lightning as pl

BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

import clipzyme.utils.loading as loaders  # noqa: E402
from clipzyme.callbacks.basic import Checkpoint  # noqa: E402

from emulator_bench import esm_cache, fast_esm, virtual_ranks  # noqa: E402

Checkpoint.CHECKPOINT_NAME_LAST = "checkpoint_last"
Checkpoint.FILE_EXTENSION = ".pt"

resume_path = os.environ.get("EMULATOR_RESUME_CKPT")
if resume_path:
    original_fit = pl.Trainer.fit

    def fit(self, *args, **kwargs):
        return original_fit(self, *args, ckpt_path=resume_path, **kwargs)

    pl.Trainer.fit = fit

if os.environ.get("EMULATOR_FAST"):
    fast_esm.install_training(always_recompute=bool(os.environ.get("EMULATOR_CHECKPOINT_ESM")))
elif os.environ.get("EMULATOR_CHECKPOINT_ESM"):
    fast_esm.recompute = True
    fast_esm.patch_layer_recompute()

if os.environ.get("EMULATOR_FROZEN_ESM"):
    esm_cache.install_frozen_esm(os.environ["EMULATOR_FROZEN_ESM"])
if os.environ.get("EMULATOR_VIRTUAL_GPUS"):
    virtual_ranks.install(int(os.environ["EMULATOR_VIRTUAL_GPUS"]))

counts = {}


def count_dataset(name, key):
    original = getattr(loaders, name)

    def wrapped(*args, **kwargs):
        loader = original(*args, **kwargs)
        counts[key] = len(loader.dataset)
        return loader

    setattr(loaders, name, wrapped)


count_dataset("get_train_dataset_loader", "train")
count_dataset("get_eval_dataset_loader", "val")
atexit.register(lambda: Path(os.environ["EMULATOR_COUNTS_PATH"]).write_text(json.dumps(counts)))

main_script = BASELINE_ROOT / "scripts" / "main.py"
sys.argv = [str(main_script), *sys.argv[1:]]
runpy.run_path(str(main_script), run_name="__main__")
