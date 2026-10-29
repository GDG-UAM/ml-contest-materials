# GDG UAM ML Contest: participant materials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GDG-UAM/ml-contest-materials/blob/main/colab.ipynb)

Classify multichannel time series into 8 classes. Your model sees one sample at a time, and the
clock is running: you are scored on how many samples you answer, and how well, within a fixed
prediction-time budget.

| File                  | What it is                                                              |
| --------------------- | ----------------------------------------------------------------------- |
| `competition_api.py`  | `Sample`, `ABSTAIN`, `N_CLASSES`, `load_dataset` and the model contract |
| `harness.py`          | the official evaluation harness: run it to test your model locally      |
| `examples/`           | a baseline: generic features + scikit-learn gradient boosting           |
| `colab.ipynb`         | the same in Google Colab: materials, data, baseline, submission check   |
| `data/train_100k.npz` | training data, 100,000 labeled samples (download it, see below)         |
| `data/test_demo.npz`  | demo test set, 400 labeled samples drawn like the evaluation data       |

## Setup

Requires Python 3.10+ and numpy (scikit-learn for the baseline).

```bash
git clone https://github.com/GDG-UAM/ml-contest-materials.git
cd ml-contest-materials
pip install numpy scikit-learn
# training data (413 MB); in Windows PowerShell write curl.exe instead of curl
curl -L --create-dirs -o data/train_100k.npz https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/train_100k.npz
# demo test set (1.7 MB)
curl -L --create-dirs -o data/test_demo.npz https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/test_demo.npz
```

Or open [`colab.ipynb`](colab.ipynb) in Google Colab (the badge above): it does all of this for
you, and runs the baseline.

Run every command below from the repository folder.

Quick check with the baseline:

```bash
python examples/baseline_train.py                                # trains on data/train_100k.npz
python harness.py evaluate --model examples/baseline_model.py    # evaluates it like the real thing
```

## Your submission

Submit a GitHub repository that contains:

- `model.py` at the top level, defining `class Model` (see below);
- your trained model: weights and any helper modules `model.py` loads;
- `requirements.txt` with the libraries `model.py` uses: we install them before the evaluation,
  since there is no internet while your model runs.

Nothing is trained on our side: `Model()` must load your already trained model from the
repository. Email the repository link to all four organizers: **gdguam@gmail.com,
ibondmingo4@gmail.com, adam.maltoni@gmail.com, hector.tablerodiaz@gmail.com**. Make sure we can
clone it (public, or shared with us if private). GitHub rejects files over 100 MB, so use
[Git LFS](https://git-lfs.com) for larger weights.

Before sending it, check that it works from a fresh clone:

```bash
git clone <your repository URL> submission-check
python harness.py evaluate --model submission-check/model.py
```

`model.py` defines `class Model`:

```python
from pathlib import Path

from competition_api import ABSTAIN, Sample

HERE = Path(__file__).parent  # load your files relative to this file


class Model:
    def __init__(self):
        # Load weights here. Untimed, but must finish within the setup timeout.
        self.weights = HERE / "weights.pt"

    def predict(self, x: Sample) -> int:
        # Return a class id in 0..7, or ABSTAIN (-1) to pass.
        return ABSTAIN
```

- `x.series` is a read-only `float32` array of shape `(4, 256)` (4 channels, 256 time steps) that may contain NaN.
- `x.meta` is a dict with `sample_rate`, `group` and `level`.
- Labels are never passed to your model.
- An exception, or any return value other than an `int` in `0..7` or `-1`, counts as a wrong answer.

**Use relative paths.** Your model runs with its own folder as the working directory, and that
folder is on `sys.path`, both locally and in the official evaluation. Keep the files it needs
(weights, helper modules) next to it or in subfolders, and build their paths from `__file__` as
above. Absolute paths such as `C:/Users/you/...` or `/home/you/...` do not exist on the evaluation
machine; the harness warns you when it finds one in your model file.

## Training data

```python
from competition_api import load_dataset

ds = load_dataset("data/train_100k.npz")   # memory-mapped, so opening it is instant
ds.series   # (100000, 4, 256) float32, may contain NaN
ds.labels   # (100000,) class ids 0..7
ds.meta     # {"sample_rate": array, "group": array, "level": array}
ds[j]       # row j as the exact Sample your model would receive
```

- The evaluation data may differ from the training data in noise, ranges and class balance. Don't
  rely on quirks of the training file.
    - PS: Mean accuracy of the baseline model is only ~0.41 in test. Remember to analyze structural differences in the data.
- `data/test_demo.npz` has 400 labeled samples drawn like the evaluation data, so you can check
  your model on them (see [Local evaluation](#local-evaluation)).

## How evaluation works

**Every model is evaluated in Google Colab**, so all teams are compared under the same conditions.
You don't need Colab to develop: any machine with Python 3.10+ works, and `harness.py` applies
the same limits there. Speeds differ between machines, though, so if your model is close to the
time limits, try it in Colab too ([`colab.ipynb`](colab.ipynb)).

Each team's model is evaluated in 3 independent runs, and the ranking uses the mean score. Each
run starts a fresh process and goes through three phases:

1. **Setup:** `Model()` is created. Untimed, but it must finish within 120 s.
2. **Warmup (unscored):** up to 100 samples or 5 s of prediction time. If a call overruns the
   warmup limits (2 s per call, or the time left of the 5 s), your model is restarted (setup runs
   again) and goes straight to the scored phase.
3. **Scored:** samples arrive one at a time. Only `predict` is timed, and the run ends once your
   total prediction time reaches **60 s**. The faster you are, the more samples you answer.
    - A call that takes longer than **0.5 s** counts as wrong (its time is still charged).
    - A call that has not returned after 0.75 s is killed, and the run ends there.

Numerical libraries are limited to one thread (`OMP_NUM_THREADS=1` and similar), and Python's `random`,
`numpy.random` and `torch` are seeded for each run. The authoritative values are in `EVAL_CONFIG`
at the top of `harness.py`.

## Scoring

```math
\text{Score} = \left(\prod_{k=0}^{7} T_k\right)^{1/8},
\qquad
T_k = \max\left(\varepsilon,\; n_k^{\alpha} \cdot \frac{c_k - \lambda w_k}{n_k} \cdot \left(\frac{c_k}{n_k}\right)^{p}\right)
```

Over the scored samples whose true class is $k$: $n_k$ is how many there were, $c_k$ how many you
got right and $w_k$ how many you got wrong. Timeouts, exceptions and invalid outputs count as
wrong. Abstentions count in $n_k$, but as neither right nor wrong.

| Parameter     | Value | Meaning                                                  |
| ------------- | ----- | -------------------------------------------------------- |
| $\varepsilon$ | 0.01  | minimum of each class term                               |
| $\lambda$     | 0.33  | penalty for each wrong answer                            |
| $\alpha$      | 0.5   | speed exponent: answering 4× more samples doubles a term |
| $p$           | 2     | per-class precision is squared                           |

Every class matters: the geometric mean drops sharply if one class term falls to $\varepsilon$.

## Local evaluation

```bash
python harness.py evaluate --model my_model.py                         # 3 runs on data/train_100k.npz
python harness.py evaluate --model a.py --model b.py --runs 1          # compare models
python harness.py evaluate --model my_model.py --set time_budget_s=20  # quicker runs
```

- Results go to `results/`: `<model>/run_<r>.json` (score, accuracy, latency, confusion matrix,
  error log), `<model>/run_<r>.worker.log` (whatever your model prints) and `leaderboard.csv`.
- Each run reads the file in a different shuffled order. `--set` changes any `EVAL_CONFIG` value
  for your local runs only.
- By default the harness evaluates on the training data, where a model that trained on it looks
  better than it is. Use it to check speed and robustness, and estimate accuracy on rows you held
  out of training:

```python
import numpy as np
from competition_api import load_dataset

ds = load_dataset("data/train_100k.npz")
val = np.sort(np.random.default_rng(0).permutation(len(ds))[:20_000])  # don't train on these rows
np.savez("data/val.npz", series=ds.series[val], labels=ds.labels[val],
         **{f"meta__{k}": v[val] for k, v in ds.meta.items()})
```

```bash
python harness.py evaluate --model my_model.py --source file:data/val.npz
```

To check your model on data drawn like the evaluation data, use the demo test set. The official
evaluation uses new samples, not these.

```bash
python harness.py evaluate --model my_model.py --source file:data/test_demo.npz
```

The run ends once the 400 samples are used up, so expect a lower score than a full 60 s run gives
(the score grows with the number of samples you answer). Use it to compare accuracy and to compare
your models with each other.

Don't edit `harness.py` or `competition_api.py`: the official evaluation uses its own copies.
If `harness.py` warns that it differs from the official version, download it again.
