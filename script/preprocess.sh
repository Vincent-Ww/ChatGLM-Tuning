/home/xiezizhe/anaconda3090/envs/wzx_glm/bin/python cover_alpaca2jsonl.py --data_path data/ks_h2h_combine_2w/resample_instruce_dialogue_combine_train.json \
    --save_path data/ks_h2h_combine_2w/resample_instruce_dialogue_combine_train.jsonl

/home/xiezizhe/anaconda3090/envs/wzx_glm/bin/python tokenize_dataset_rows.py --jsonl_path data/ks_h2h_combine_2w/resample_instruce_dialogue_combine_train.jsonl \
    --save_path data/ks_h2h_combine_2w/ --max_seq_length 1024 --skip_overlength True
