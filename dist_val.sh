CONFIG=CVPRMedSAMLite.py #CVPRMedSAMRepViTm11.py CVPRMedSAMRepViTm15.py  DistillRepViT-ViTB_PreComputed.py DistillRepViT-LiteMedSAM.py
GPUS=4
PORT=${PORT:-29503}
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
    # --nnodes=$NNODES \
    # --node_rank=$NODE_RAN[K \PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --master_addr=$MASTER_ADDR \
    --use_env \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/val.py \
    $CONFIG 
    # --seed 0 \
    # --launcher pytorch ${@:3}
