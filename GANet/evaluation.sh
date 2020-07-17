CUDA_VISIBLE_DEVICES=0 python evaluation.py --crop_height=528 \
                  --crop_width=672 \
                  --max_disp=192 \
                  --data_path='./cmu_rectified/training/' \
                  --test_list='lists/file.list' \
                  --save_path='./result/' \
                  --resume='./checkpoint/kitti2015_final.pth' \
                  --threshold=3.0 \
                  --kitti2015=1
# 2>&1 |tee logs/log_evaluation.txt
exit