CONFIG=MedSAMEncoder1024.py
GPUS=3
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

    # distributed.launch parameters I commented out
    # --nnodes=$NNODES \
    # --node_rank=$NODE_RANK \

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/inference.py \
    $CONFIG \
    --seed 0 \
    # --launcher pytorch ${@:3}
