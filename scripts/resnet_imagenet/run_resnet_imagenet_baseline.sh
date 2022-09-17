
IMAGENET_PATH="C:/ImageNet"
PRUNE_DIR="Baseline_ImageNet"

train_baseline_resnet_imagenet()
{
    python main_resnet_imagenet.py --data_path $IMAGENET_PATH \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --pretrain_path $4 \
    --start_epoch $5 \
    --total_epoches 100 \
    --decay_epoch_step 30 \
    --workers 6 \
    --lr 0.1 --decay 0.0001 --batch_size 256
}

#resnet18, flop 0.42
#train_baseline_resnet_imagenet train resnet18 $PRUNE_DIR/ResNet18_0.42 $PRUNE_DIR/ResNet18_0.42/resnet18.epoch.25.pth.tar 25

#resnet18, flop 0.45
#train_baseline_resnet_imagenet train resnet18 $PRUNE_DIR/ResNet18_0.45 $PRUNE_DIR/ResNet18_0.45/resnet18.epoch.45.pth.tar 45

#resnet34, flop 0.41
#train_baseline_resnet_imagenet train resnet34 $PRUNE_DIR/ResNet34_0.41 $PRUNE_DIR/ResNet34_0.41/resnet34.epoch.25.pth.tar 25

#resnet34, flop 0.45
#train_baseline_resnet_imagenet train resnet34 $PRUNE_DIR/ResNet34_0.45 $PRUNE_DIR/ResNet34_0.45/resnet34.epoch.25.pth.tar 25

#resnet50, flop 0.42
#train_baseline_resnet_imagenet train resnet50 $PRUNE_DIR/ResNet50_0.42 $PRUNE_DIR/ResNet50_0.42/resnet50.epoch.40.pth.tar 40

#resnet50, flop 0.54
train_baseline_resnet_imagenet train resnet50 $PRUNE_DIR/ResNet50_0.54 $PRUNE_DIR/ResNet50_0.54/resnet50.checkpoint.pth.tar 40