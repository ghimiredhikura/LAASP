
IMAGENET_PATH="C:/ImageNet"
EVAL_DIR="ImageNet_EVAL"

eval_resnet_imagenet()
{
    python main_resnet_imagenet.py --data_path $IMAGENET_PATH \
    --mode $1 \
    --arch $2 \
    --save_path $3 \
    --baseline_path $4 \
    --pruned_path $5 
    --workers 8 \
    --batch_size 256
}

# resnet18 
BASELINE_PATH="models_baseline/ImageNet/ResNet18/resnet18.model_best.pth.tar"
PRUNED_PATH="models_pruned/ImageNet/ResNet18_FlopRed_0.42/resnet18.model_best.pth.tar"
eval_resnet_imagenet eval resnet18 $EVAL_DIR/resnet18_0.42 $BASELINE_PATH $PRUNED_PATH


# resnet34 
BASELINE_PATH="models_baseline/ImageNet/ResNet34/resnet34.model_best.pth.tar"
PRUNED_PATH="models_pruned/ImageNet/ResNet34_FlopRed_0.41/resnet34.model_best.pth.tar"
eval_resnet_imagenet eval resnet34 $EVAL_DIR/resnet34_0.41 $BASELINE_PATH $PRUNED_PATH


# resnet50 
BASELINE_PATH="models_baseline/ImageNet/ResNet50/resnet50.model_best.pth.tar"
PRUNED_PATH="models_pruned/ImageNet/ResNet50_FlopRed_0.42/resnet50.model_best.pth.tar"
eval_resnet_imagenet eval resnet50 $EVAL_DIR/resnet50_0.42 $BASELINE_PATH $PRUNED_PATH