#/home/xiezizhe/anaconda3090/envs/wzx_glm/bin/python cover_alpaca2jsonl.py --data_path data/ks_h2h_question/resample_h2h_question_train.json \
#    --save_path data/ks_h2h_question/resample_h2h_question_train.jsonl

#/home/xiezizhe/anaconda3090/envs/wzx_glm/bin/python tokenize_dataset_rows.py --jsonl_path data/ks_h2h_question/resample_h2h_question_train.jsonl \
#    --save_path data/ks_h2h_question/ --max_seq_length 1024 --skip_overlength True

#/home/xiezizhe/anaconda3090/envs/wzx_glm/bin/python cover_alpaca2jsonl.py --data_path data/ks_h2h_combine/resample_h2h_combine_train.json \
#     --save_path data/ks_h2h_combine/resample_h2h_combine_train.jsonl

#/home/xiezizhe/anaconda3090/envs/wzx_glm/bin/python tokenize_dataset_rows.py --jsonl_path data/ks_h2h_combine/resample_h2h_combine_train.jsonl \
#     --save_path data/ks_h2h_combine/ --max_seq_length 1024 --skip_overlength True

# 智能对话+FT
#/home/xiezizhe/anaconda3090/envs/wzx_glm/bin/python cover_alpaca2jsonl.py --data_path data/ks_ai_ft3/ks_ai_nomanual_ft_format_12-25_train.json \
#     --save_path data/ks_ai_ft3/ks_ai_nomanual_ft_format_12-25_train.jsonl  --data_size 50000

#/home/xiezizhe/anaconda3090/envs/wzx_glm/bin/python tokenize_dataset_rows.py --jsonl_path data/ks_ai_ft3/ks_ai_nomanual_ft_format_12-25_train.jsonl \
#     --save_path data/ks_ai_ft3/ --max_seq_length 1024 --skip_overlength True

# 智能对话+FT（有标注数据）
/home/xiezizhe/anaconda3090/envs/wzx_glm/bin/python cover_alpaca2jsonl.py --data_path data/ks_ai_ft_anno/ks_ai_ft_0519-0601_anno_train.json \
     --save_path data/ks_ai_ft_anno/ks_ai_ft_0519-0601_anno_train.jsonl  --data_size 50000

/home/xiezizhe/anaconda3090/envs/wzx_glm/bin/python tokenize_dataset_rows.py --jsonl_path data/ks_ai_ft_anno/ks_ai_ft_0519-0601_anno_train.jsonl \
     --save_path data/ks_ai_ft_anno/ --max_seq_length 1024 --skip_overlength True
