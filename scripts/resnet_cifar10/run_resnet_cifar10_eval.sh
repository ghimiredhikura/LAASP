
EVAL_DIR="CIFAR10_ResNet_EVAL"

eval_cifar10_resnet()
{
    python main_resnet_cifar10.py --data_path ./data/cifar.python --dataset cifar10 \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --baseline_path $4 \
    --pruned_path $5 
}

# eval resnet32 

NET="resnet32"
BASELINE_PATH="models_baseline/CIFAR10/ResNet/ResNet32/ResNet32.1.0.53/resnet32.model_best.pth.tar"
PRUNED_PATH="models_pruned/CIFAR10/ResNet/ResNet32/resnet32.1.0.53/resnet32.model_best.pth.tar"

eval_cifar10_resnet eval $NET $EVAL_DIR/$NET $BASELINE_PATH $PRUNED_PATH