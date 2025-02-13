# make sure directory is in the right place and add it to Python path in order to find package
cd "/share/colon/cclaessens/ovacadx" || exit
export HOME="/share/colon/cclaessens/"
export PYTHONPATH="${PYTHONPATH}:/share/colon/cclaessens/ovacadx"

# login weights and biases and set ouput directory
mkdir -p "/share/colon/cclaessens/ovacadx/wandb"
# make sure to set WANDB_API_KEY in environment variables or add it here as
# export WANDB_API_KEY=<your_wandb_api_key>
# do not publish your API key to public repositories!
export WANDB_DIR="/share/colon/cclaessens/ovacadx/wandb/"
export WANDB_CONFIG_DIR="/share/colon/cclaessens/ovacadx/wandb/"
export WANDB_CACHE_DIR="/share/colon/cclaessens/ovacadx/wandb/"
export WANDB_START_METHOD="thread"
wandb login

python3 experiments/pretraining/pretrain_volume_encoder.py \
    --data_path "/share/colon/cclaessens/datasets/AbdomenCT-1K" \
    --mil_mode "att_trans" \
    --pretrained_path "checkpoints/pretraining/slice_encoder/resnet34/model-epoch=099-val_loss=0.00.ckpt" \
    --backbone "resnet34" \
    --batch_size 4 \
    --drop_last_batch \
    --pin_memory \
    --nodes 1 \
    --gpus 2
