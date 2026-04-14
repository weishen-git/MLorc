# MLorc: Momentum Low-rank Compression for Memory-Efficient LLM Adaptation

This repository provides the official implementation for the AISTATS 2026 paper:

> **MLorc: Momentum Low-rank Compression for Memory-Efficient Large Language Model Adaptation**

---

## Overview

MLorc is a memory-efficient optimizer framework for fine-tuning large language models (LLMs). Instead of maintaining full-rank first- and second-moment matrices as in standard Adam-style optimizers, MLorc applies **randomized SVD** to compress optimizer states into low-rank factored representations — drastically reducing memory overhead without requiring changes to the model architecture.


### Key Features

- **Low-rank optimizer states**: Both the first moment (momentum) and second moment (variance) are stored as rank-`r` factorizations `U @ diag(S) @ V`, where `r` is much smaller than the weight matrix dimensions.
- **Randomized SVD**: Efficient approximate SVD via random projections, with optional PyTorch API backend (`torch.pca_lowrank`).
- **Two optimizers**: `MLorc_AdamW` (Adam with bias correction) and `MLorc_Lion` (sign-based update rule).



## Install experiment dependencies

```bash
pip install -r exp_requirements.txt
```


---

## Usage

### Core Optimizers

```python
from MLorc.MLorc_AdamW import MLorc_AdamW
from MLorc.MLorc_Lion import MLorc_Lion


optimizer = MLorc_AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    rank=4,          # low-rank approximation rank
)


optimizer = MLorc_Lion(
    model.parameters(),
    lr=1e-4,
    betas=(0.95, 0.98),
    weight_decay=0.05,
    rank=4          # low-rank approximation rank
)
```

> **Note:** MLorc only compresses optimizer states for 2D parameter tensors (weight matrices).


---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
shen2026mlorc,
title={{ML}orc: Momentum Low-rank Compression for Memory Efficient Large Language Model Adaptation},
author={Wei Shen and Zhang Yaxiang and Minhui Huang and Mengfan Xu and Jiawei Zhang and Cong Shen},
booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
year={2026},
url={https://openreview.net/forum?id=sw7gHBmXls}
}
```
