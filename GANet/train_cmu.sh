#Fine tuning for kitti 2015
CUDA_VISIBLE_DEVICES=0 python train.py --batchSize=16 \
                --crop_height=528 \
                --crop_width=672 \
                --max_disp=32 \
                --thread=8 \
                --data_path='./CMU/training/' \
                --training_list='lists/CMU.list' \
                --save_path='./checkpoint/finetune_kitti2015' \
                --kitti2015=1 \
                --shift=3 \
                --nEpochs=800 2>&1 |tee logs/log_finetune_kitti2015.txt

CUDA_VISIBLE_DEVICES=0 python train.py --batchSize=8 \
                --crop_height=528 \
                --crop_width=672 \
                --max_disp=32 \
                --thread=8 \
                --data_path='./CMU/training/' \
                --training_list='lists/CMU.list' \
                --save_path='./checkpoint/finetune2_kitti2015' \
                --kitti2015=1 \
                --shift=3 \
                --lr=0.0001 \
                --resume='./checkpoint/finetune_kitti2015_epoch_800.pth' \
                --nEpochs=8 2>&1 |tee logs/log_finetune_kitti2015.txt