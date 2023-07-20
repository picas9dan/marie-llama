## Installation

### Prerequisites
- Linux OS (recommended) . It is not advisable to run this project on Windows as [`bitsandbytes` is not supported on Windows]((https://github.com/TimDettmers/bitsandbytes/issues/30)).
- [conda](https://conda.io/projects/conda/en/latest/index.html) (recommended).
- `python==3.10`.

### Steps
1. Create a conda environment and activate it.
   ```
   conda create --name marie-llama python=3.10
   conda activate marie-llama
   ```
1. Install `torch==2.0.1+cu118`.
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
1. Install the remaining packages required by the project.

   ````
   pip install -r requirements.txt
   ````
1. For access to any private models or datasets on Hugging Face, ensure that you have the necessary read privileges. Generate an [access token](https://huggingface.co/docs/hub/security-tokens) and make this an environment variable.
   ```
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


## Running jobs on CSD3
Per [recommendation by CSD3](https://docs.hpc.cam.ac.uk/hpc/user-guide/io_management.html), I/O data should be placed under `/rds`. Cache for models and datasets should thus be placed here. Concretely, the Hugging Face cache directory should be set as follows.
```
export HF_HOME="/rds/user/nmdt2/hpc-work/.cache/huggingface"
``` 
