## Installation

The use of a virtual environment manager such as [conda](https://conda.io/projects/conda/en/latest/index.html) is recommended.

Prerequisites
- Linux OS
  - As per [this issue](https://github.com/TimDettmers/bitsandbytes/issues/30), the bitsandbytes package is not supported on Windows.
- python==3.10
- [torch](https://pytorch.org/get-started/locally/)==2.0.1+cu117

Install other dependencies with the following command.

```
pip install -r requirements
```

## Data Generation/Preparation

## Inference

### py

```
python
    --model_name_or_path <>

```

### cpp

## Fine-tuning



```
python finetune.py
    --per_device_train_batch_size
```

## Evaluation
