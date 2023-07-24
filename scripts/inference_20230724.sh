python inference.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --lora_adapter_dir /rds/user/nmdt2/hpc-work/outputs/llama-2-7b/checkpoint-90/adapter_model \
    --eval_data_path ./data/test_20230721.json \
    --prompt_template marie_no_context \
    --source_max_len 512 \
    --target_max_len 512
