seed=${RANDOM}
CUDA_VISIBLE_DEVICES=0 python main.py \
--image_dir data/mimic_cxr/images/ \
--ann_path data/mimic_cxr/annotation.json \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--batch_size 16 \
--epochs 40 \
--save_dir  \
--step_size 1 \
--d_vf 2048 \
--gamma 0.8 \
--seed ${seed} \
--early_stop 40 
