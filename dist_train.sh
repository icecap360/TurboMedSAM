CONFIG=DistillRepVit.py
GPUS=3
PORT=${PORT:-29500}
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
    # --nnodes=$NNODES \
    # --node_rank=$NODE_RANK \PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    # --launcher pytorch ${@:3}
