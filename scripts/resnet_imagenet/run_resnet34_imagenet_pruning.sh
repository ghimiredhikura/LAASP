
IMAGENET_PATH="C:/ImageNet"
PRUNE_DIR="C:/Deepak/ImageNet_PRUNE_OneShot"

train_prune_baseline_resnet_imagenet()
{
    python main_resnet_imagenet_PC4.py --data_path $IMAGENET_PATH \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --rate_flop $4 \
    --total_epoches 100 \
    --recover_epoch 2 \
    --prune_epoch 25 \
    --decay_epoch_step 30 \
    --workers 8 \
    --lr 0.1 --decay 0.0001 --batch_size 256
}

train_prune_baseline_resnet_imagenet prune resnet34 $PRUNE_DIR/resnet34_0.45 0.45