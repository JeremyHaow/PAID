#!/bin/bash

# MSTD 模型运行脚本
# 提供简单的命令行界面来运行 MSTD 模型

# 显示帮助信息
show_help() {
    echo "MSTD AI生成图像检测器运行脚本"
    echo ""
    echo "用法: ./run_mstd.sh [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  train         训练新模型"
    echo "  eval          在验证集上评估模型"
    echo "  test          在测试集上测试模型"
    echo ""
    echo "选项:"
    echo "  --gpu NUM     指定使用的GPU编号 (默认: 0)"
    echo "  --batch INT   指定批处理大小 (默认: 32)"
    echo "  --epochs INT  指定训练轮数 (默认: 20)"
    echo "  --lr FLOAT    指定学习率 (默认: 1e-4)"
    echo "  --model STR   指定CLIP模型类型 (默认: ViT-L/14)"
    echo "  --ckpt PATH   指定检查点路径 (用于评估/测试)"
    echo "  --config PATH 指定配置文件路径"
    echo "  --data PATH   指定数据根目录"
    echo "  --adv         启用对抗训练"
    echo "  --high-res    使用高分辨率设置 (384x384)"
    echo ""
    echo "示例:"
    echo "  ./run_mstd.sh train --gpu 0 --batch 64 --epochs 30"
    echo "  ./run_mstd.sh eval --ckpt checkpoints/best_model.pth"
    echo "  ./run_mstd.sh test --ckpt checkpoints/best_model.pth --high-res"
    echo ""
}

# 如果没有参数，显示帮助
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# 解析第一个参数作为命令
COMMAND=$1
shift

# 默认参数值
GPU=0
BATCH_SIZE=32
EPOCHS=20
LEARNING_RATE=0.0001
MODEL="ViT-L/14"
CHECKPOINT=""
CONFIG_FILE=""
DATA_ROOT=""
IMAGE_SIZE=224
EXTRA_ARGS=""

# 解析选项
while [[ $# -gt 0 ]]; do
    key="$1"
    
    case $key in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --ckpt)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data)
            DATA_ROOT="$2"
            shift 2
            ;;
        --adv)
            EXTRA_ARGS="$EXTRA_ARGS --adv_epsilon 0.02 --adv_alpha 0.8"
            shift
            ;;
        --high-res)
            IMAGE_SIZE=384
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知选项 '$key'"
            show_help
            exit 1
            ;;
    esac
done

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=$GPU

# 构建基本命令
CMD="python main.py --mode $COMMAND --batch_size $BATCH_SIZE --num_epochs $EPOCHS --learning_rate $LEARNING_RATE --base_model $MODEL --image_size $IMAGE_SIZE"

# 添加额外参数
if [ ! -z "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

if [ ! -z "$CONFIG_FILE" ]; then
    CMD="$CMD --config_file $CONFIG_FILE"
fi

if [ ! -z "$DATA_ROOT" ]; then
    CMD="$CMD --train_data_path $DATA_ROOT/train --val_data_path $DATA_ROOT/val --test_data_path $DATA_ROOT/test"
fi

# 添加任何额外参数
CMD="$CMD $EXTRA_ARGS"

# 执行命令
echo "执行命令: $CMD"
eval $CMD

# 检查命令是否成功
if [ $? -eq 0 ]; then
    echo "完成!"
else
    echo "失败! 请检查错误信息。"
fi
