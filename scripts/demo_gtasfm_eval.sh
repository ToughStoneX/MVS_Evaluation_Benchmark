PCS_PATH="outputs/gtasfm/eval/"
# [NOTE] Modify "GTS_PATH" to the path of ground truth files saved in your machine
GTS_PATH="/home/xhb/datasets/evaluation_gtasfm/Points/stl"

CUDA_VISIBLE_DEVICES=2 python eval_gtasfm.py \
    --prefix "mvsnet" \
    --pcs_path ${PCS_PATH} \
    --gts_path ${GTS_PATH}


    