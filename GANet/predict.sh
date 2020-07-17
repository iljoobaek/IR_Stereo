
python predict.py --crop_height=528 \
                  --crop_width=672 \
                  --max_disp=64 \
                  --data_path='./cmu_rectified/training/' \
                  --test_list='lists/file.list' \
                  --save_path='./result/' \
                  --kitti2015=1 \
                  --resume='./checkpoint/kitti2015_final.pth'
exit