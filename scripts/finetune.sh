python finetune.py \
    --model_path google/flan-t5-base \
    --data_path ./data/train.json \
    --output_dir /rds/user/nmdt2/hpc-work/outputs \
    --do_train \
    --per_device_train_batch_size 16 \
    --num_train_epochs 3 \
    --optim paged_adamw_32bit \
    --learning_rate 0.0002 \
    --logging_steps 1