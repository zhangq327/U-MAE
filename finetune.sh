

CUDA_VISIBLE_DEVICES=8 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 main_linprobe.py \
    --accum_iter 1 \
    --batch_size 256 \
    --model vit_base_patch16 --cls_token\
    --finetune output_dir/baseline-200-base/checkpoint-180.pth \
    --epochs 90 \
    --blr 0.1  \
    --weight_decay 0.0 \
    --log_dir temp_dir\
    --dist_eval --data_path data/imagenet100
