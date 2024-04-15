#CUDA_VISIBLE_DEVICES=2,6 \
torchrun --nproc_per_node=4  train_downstream.py \
    --Train_csv_path /data/workspace/zhangqingchuan/medllm/PMC-VQA/MedVInT_TD/Data/Slake1.0/train.csv \
    --Eval_csv_path /data/workspace/zhangqingchuan/medllm/PMC-VQA/MedVInT_TD/Data/Slake1.0/val.csv \
    --output_dir /data/workspace/zhangqingchuan/medllm/PMC-VQA/MedVInT_TD/Results/QA_no_pretrain_no_aug \
    # --Train_csv_path /data/workspace/zhangqingchuan/medllm/PMC-VQA/MedVInT_TD/Data/VQA_RAD/train.csv \
    # --Eval_csv_path /data/workspace/zhangqingchuan/medllm/PMC-VQA/MedVInT_TD/Data/VQA_RAD/val.csv \

    --run_name QA_no_pretrain_no_aug \
    --num_train_epochs 100 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ./ds_config/ds_config_zero2.json \
    --checkpointing false \
    --bf16 True \
    # --tf32 True
