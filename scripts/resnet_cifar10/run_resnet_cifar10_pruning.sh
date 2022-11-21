
PRUNE_DIR="CIFAR10_ResNet_PRUNE"

pruning_ratio_pretrain_cifar10_resnet()
{
    python main_resnet_cifar10.py --data_path ./data/cifar.python --dataset cifar10 \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --rate_flop $4 \
    --total_epoches 200 \
    --schedule 60 120 160 190 \
    --gammas 0.2 0.2 0.2 0.2 \
    --recover_flop 0.0 \
    --recover_epoch 1 \
    --prune_epoch $5 \
    --workers 8 \
    --lr 0.1 --decay 0.0005 --batch_size 256
}

run_110()
{
    NET="resnet110"
    FLOP_RATE="0.522"

    PRUNE_EPOCH_NO=50
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.1.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.2.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.3.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
}

#run_110

run_56()
{
    NET="resnet56"
    FLOP_RATE="0.525"

    PRUNE_EPOCH_NO=50
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.1.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.2.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.3.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
}

#run_56

run_32()
{
    NET="resnet32"
    FLOP_RATE="0.53"

    PRUNE_EPOCH_NO=50
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.1.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.2.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
    pruning_ratio_pretrain_cifar10_resnet prune $NET $PRUNE_DIR/$PRUNE_EPOCH_NO.$NET.3.$FLOP_RATE $FLOP_RATE $PRUNE_EPOCH_NO
}

run_32