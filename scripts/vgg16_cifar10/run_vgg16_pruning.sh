
BASELINE_PATH="BASELINE/VGG16_BASELINE_2" 
PRUNE_PATH="C:/Deepak/VGG_PRUNE"

prune_baseline_vgg16()
{
    python main_vgg_cifar10.py --dataset cifar10 --depth 16 \
    --mode prune \
    --pretrain_path $1 \
    --save_path $2 \
    --rate_flop $3 \
    --total_epoches 160 \
    --schedule 40 80 120 \
    --gammas 0.2 0.2 0.2 \
    --recover_epoch 5 \
    --lr 0.1 --decay 0.0005 --batch_size 256
}

prune_baseline_vgg16 $BASELINE_PATH/vgg16_base_1/model_best.pth.tar $PRUNE_PATH/model1_0.342 0.342
prune_baseline_vgg16 $BASELINE_PATH/vgg16_base_2/model_best.pth.tar $PRUNE_PATH/model2_0.342 0.342
prune_baseline_vgg16 $BASELINE_PATH/vgg16_base_3/model_best.pth.tar $PRUNE_PATH/model3_0.342 0.342

prune_baseline_vgg16 $BASELINE_PATH/vgg16_base_1/model_best.pth.tar $PRUNE_PATH/model1_0.50 0.50
prune_baseline_vgg16 $BASELINE_PATH/vgg16_base_2/model_best.pth.tar $PRUNE_PATH/model2_0.50 0.50
prune_baseline_vgg16 $BASELINE_PATH/vgg16_base_3/model_best.pth.tar $PRUNE_PATH/model3_0.50 0.50