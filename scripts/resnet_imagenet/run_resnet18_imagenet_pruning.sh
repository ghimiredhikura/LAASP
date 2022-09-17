
IMAGENET_PATH="C:/ImageNet"
PRUNE_DIR="C:/Deepak/ImageNet_PRUNE_OneShot"
PRETRAIN_PATH="C:/Deepak/ImageNet_PRUNE_OneShot/resnet18.epoch.25.pth.tar"
PRUNE_EPOCH="45"

train_prune_baseline_resnet_imagenet()
{
    python main_resnet_imagenet_pc3.py --data_path $IMAGENET_PATH \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --rate_flop $4 \
    --pretrain_path $PRETRAIN_PATH \
    --start_epoch 25 \
    --total_epoches 100 \
    --recover_epoch 2 \
    --prune_epoch $PRUNE_EPOCH \
    --decay_epoch_step 30 \
    --workers 8 \
    --lr 0.1 --decay 0.0001 --batch_size 256
}

train_prune_baseline_resnet_imagenet prune resnet18 $PRUNE_DIR/$PRUNE_EPOCH.resnet18_0.453 0.453