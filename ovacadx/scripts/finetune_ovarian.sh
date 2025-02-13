# make sure directory is in the right place and add it to Python path in order to find package
cd "/share/colon/cclaessens/ovacadx" || exit
export HOME="/share/colon/cclaessens/"
export PYTHONPATH="${PYTHONPATH}:/share/colon/cclaessens/ovacadx"

# login weights and biases and set ouput directory
mkdir -p "/share/colon/cclaessens/ovacadx/wandb"
export WANDB_DIR="/share/colon/cclaessens/ovacadx/wandb/"
export WANDB_CONFIG_DIR="/share/colon/cclaessens/ovacadx/wandb/"
export WANDB_CACHE_DIR="/share/colon/cclaessens/ovacadx/wandb/"
export WANDB_START_METHOD="thread"
wandb login

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 0 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 1 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 2 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 3 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 4 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 5 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 6 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 7 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 8 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 9 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 10 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 11 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 12 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 13 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 14 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 15 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 16 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 17 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 18 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 19 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 20 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 21 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 22 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 23 \
    --seed 0 \

python3 experiments/downstream/finetune_ovarian.py \
    --data_path "/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv" \
    --label "label" \
    --orient \
    --mask_tumor \
    --crop_tumor \
    \
    --backbone "resnet34" \
    --backbone_num_features 2048 \
    --pretrained_path "/share/colon/cclaessens/ovacadx/checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --mil_mode "att_trans" \
    \
    --do_rotate \
    --do_translate \
    --do_scale \
    \
    --batch_size 12 \
    --num_workers 10 \
    --max_epochs 100 \
    --lr 0.000001 \
    --num_inner_splits 5 \
    --num_outer_splits 5 \
    --k 24 \
    --seed 0 \