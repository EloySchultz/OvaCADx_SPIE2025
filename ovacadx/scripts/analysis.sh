# make sure directory is in the right place and add it to Python path in order to find package
cd "/share/colon/cclaessens/ovacadx" || exit
export HOME="/share/colon/cclaessens/"
export PYTHONPATH="${PYTHONPATH}:/share/colon/cclaessens/ovacadx"

export DATAPATH="/share/colon/cclaessens/ovacadx/checkpoints/downstream/att/resnet34/results.csv"

python3 experiments/analysis/bootstrapping.py \
    --data_path "${DATAPATH}" \
    --num_bootstraps 1000

python3 experiments/analysis/ensemble.py \
    --data_path "${DATAPATH}"