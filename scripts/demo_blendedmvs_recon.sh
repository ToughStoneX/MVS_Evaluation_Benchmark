# [NOTE] Modify "BLENDEDMVS_PATH" to the path of BlendedMVS dataset saved in your machine
BLENDEDMVS_PATH="/home/xhb/datasets/blendedmvs/"
TESTLIST="lists/blendedmvs/test.txt"
OUTDIR="./outputs/blendedmvs/"
CKPT_FILE="checkpoints/demo.ckpt"
DATASET_NAME="general_eval_blendedmvs"

printf "testing $CKPT_FILE\n"

# 3D reconstruction on the evaluation set of Blendedmvs set
CUDA_VISIBLE_DEVICES=2 python testing/test.py \
    --dataset=$DATASET_NAME --batch_size=1 \
    --testpath=$BLENDEDMVS_PATH  --testlist=$TESTLIST \
    --loadckpt $CKPT_FILE --outdir $OUTDIR  --interval_scale 1.06  --filter_method gipuma \
    --max_h 576 --max_w 768 \
    --num_consistent 3 --prob_threshold 0.4 --disp_threshold 0.13

# Moving the generated point clouds (.ply) to the same directory (outputs/blendedmvs/eval) and prepare for evaluation
python scripts/arange_blendedmvs.py