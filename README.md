## Installation

The use of a virtual environment manager such as [conda](https://conda.io/projects/conda/en/latest/index.html) is recommended.

### Prerequisites
- Linux OS
  - As per [this issue](https://github.com/TimDettmers/bitsandbytes/issues/30), the bitsandbytes package is not supported on Windows.
- `python==3.10`

### Steps
1. Install `torch==2.0.1+cu117` as per [official documentation](https://pytorch.org/get-started/locally/).
1. Install the remaining packages required by the project.

   ````
   pip install -r requirements
   ````
1. Update the following environment variables.
   - `HF_HOME`: Cache directory for Hugging Face models and datasets. For jobs running on CSD3, this should be set to `/home/<CRSid>/rds/hpc-work/.cache/huggingface`
   - `HF_ACCESS_TOKEN`: [Access token](https://huggingface.co/docs/hub/security-tokens) with the read privilege to access any model that is not public on Hugging Face, if needed.
   ```
   export HF_HOME="<huggingface-cache-dir>"
   export HF_ACCESS_TOKEN="<huggingface-access-token>"
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
