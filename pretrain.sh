python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path imagenet\
    --lambda1 0.0001\
    --reg spectral\
    --output_dir temp_dir\
    --log_dir temp_dir\
    --tau 0.1\
    # --distributed
