PCS_PATH="outputs/blendedmvs/eval/"
# [NOTE] Modify "GTS_PATH" to the path of ground truth files saved in your machine
GTS_PATH="/home/xhb/datasets/evaluation_blendedmvs/Points/stl"

CUDA_VISIBLE_DEVICES=2 python eval_blendedmvs.py \
    --prefix "mvsnet" \
    --pcs_path ${PCS_PATH} \
    --gts_path ${GTS_PATH}