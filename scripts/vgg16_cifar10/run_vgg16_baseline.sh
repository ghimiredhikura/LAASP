
train_baseline_vgg16()
{
    python vgg_cifar10_pruning.py --dataset cifar10 --depth 16 \
    --mode train \
    --save_path $1 \
    --total_epoches 160 \
    --schedule 40 80 120 \
    --gammas 0.2 0.2 0.2 \
    --lr 0.1 --decay 0.0005 --batch_size 256
}

eval_baseline_vgg16()
{
    python vgg_cifar10.py --dataset cifar10 --depth 16 \
    --mode eval \
    --pretrain_path $1 \
    --save_path $2
}

run_train_baseline_vgg16()
{
    train_baseline_vgg16 ./vgg16_baseline/vgg16_base_1
    train_baseline_vgg16 ./vgg16_baseline/vgg16_base_2
    train_baseline_vgg16 ./vgg16_baseline/vgg16_base_3
}

run_eval_baseline_vg16()
{
    eval_baseline_vgg16 ./vgg16_baseline/vgg16_base_1/model_best.pth.tar ./vgg16_eval/vgg16_base_1
    eval_baseline_vgg16 ./vgg16_baseline/vgg16_base_2/model_best.pth.tar ./vgg16_eval/vgg16_base_2
    eval_baseline_vgg16 ./vgg16_baseline/vgg16_base_3/model_best.pth.tar ./vgg16_eval/vgg16_base_3
}

run_eval_baseline_vg16