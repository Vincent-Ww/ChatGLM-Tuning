python finetune.py \
    --dataset_path /home/xiezizhe/wuzixun/LLM/ChatGLM-Tuning/data/ks_h2h_question \
    --lora_rank 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 12 \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 10 \
    --output_dir output-question-20230520
