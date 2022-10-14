
pip install torch==1.7.1
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install submitit
pip install tensorboard


python submitit_pretrain.py \
    --nodes 1 \
    --ngpus 8 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /data/imagenet