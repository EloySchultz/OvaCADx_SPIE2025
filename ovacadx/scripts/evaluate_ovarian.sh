# make sure directory is in the right place and add it to Python path in order to find package
cd "/share/colon/cclaessens/ovacadx" || exit
export HOME="/share/colon/cclaessens/"
export PYTHONPATH="${PYTHONPATH}:/share/colon/cclaessens/ovacadx"

python3 experiments/downstream/evaluate_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --ckpt_dir "checkpoints/downstream/att_trans_pyramid/resnet34/pretrained/nested-cross-validation-2-with-outliers" \
    --mil_mode "att_trans_pyramid" \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --seed 0 \

python3 experiments/downstream/evaluate_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --ckpt_dir "checkpoints/downstream/att_trans_pyramid/resnet34/" \
    --mil_mode "att_trans_pyramid" \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --seed 0 \

python3 experiments/downstream/evaluate_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --ckpt_dir "checkpoints/downstream/att_trans/resnet34/pretrained" \
    --mil_mode "att_trans" \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --seed 0 \

python3 experiments/downstream/evaluate_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --ckpt_dir "checkpoints/downstream/att_trans/resnet34/" \
    --mil_mode "att_trans" \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --seed 0 \

    python3 experiments/downstream/evaluate_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --ckpt_dir "checkpoints/downstream/att/resnet34/" \
    --mil_mode "att" \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --seed 0 \