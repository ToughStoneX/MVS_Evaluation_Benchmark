# [NOTE] Modify "GTASFM_PATH" to the path of GTASFM dataset saved in your machine
GTASFM_PATH="/home/xhb/datasets/gtasfm/gta_sfm_clean/"
TESTLIST="lists/gtasfm/test.txt"
OUTDIR="./outputs/gtasfm/"
CKPT_FILE="checkpoints/demo.ckpt"
DATASET_NAME="general_eval_gtasfm"

printf "testing $CKPT_FILE\n"

# 3D reconstruction on the test set of GTASFM set
CUDA_VISIBLE_DEVICES=2 python testing/test.py \
    --dataset=$DATASET_NAME --batch_size=1 \
    --testpath=$GTASFM_PATH  --testlist=$TESTLIST \
    --loadckpt $CKPT_FILE --outdir $OUTDIR  --interval_scale 1.06  --filter_method gipuma \
    --max_h 480 --max_w 640 \
    --num_consistent 3 --prob_threshold 0.4 --disp_threshold 0.13

# Moving the generated point clouds (.ply) to the same directory (outputs/gtasfm/eval) and prepare for evaluation
python scripts/arange_gtasfm.py