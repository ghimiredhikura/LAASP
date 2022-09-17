
IMAGENET_PATH="C:/ImageNet"
PRUNE_DIR="C:/Deepak/ImageNet_PRUNE_OneShot"

train_prune_baseline_resnet_imagenet()
{
    python main_resnet_imagenet_pc3.py --data_path $IMAGENET_PATH \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --rate_flop $4 \
    --pretrain_path "C:/Deepak/ImageNet_PRUNE_OneShot/40.resnet50_0.42/resnet50.checkpoint.pth.tar" \
    --total_epoches 100 \
    --recover_epoch 2 \
    --prune_epoch 40 \
    --decay_epoch_step 30 \
    --workers 8 \
    --lr 0.1 --decay 0.0001 --batch_size 128
}

train_prune_baseline_resnet_imagenet prune resnet50 $PRUNE_DIR/40.resnet50_0.42 0.42