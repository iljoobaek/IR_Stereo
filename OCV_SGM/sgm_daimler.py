import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

file_path = "KITTI"

disp_path = "disp_gt"
left_path_list = []
right_path_list = []
disp_path_list = []

def data_loader():
    for image_left in os.listdir(left_path):
        a = image_left.split("t")
        l_img_path = os.path.join(left_path, image_left)
        image_right = "imgright"+a[1]
        r_img_path = os.path.join(right_path, image_right)
        image_disp = "disp"+a[1]
        disp_img_path = os.path.join(disp_path, image_disp)
        left_path_list.append(l_img_path) 
        right_path_list.append(r_img_path)
        disp_path_list.append(disp_img_path)
        
def kitti_loader():
    for 
    filename = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
    left = Image.open(filename)
    filename = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
    right = Image.open(filename)
    filename = file_path + 'disp_occ_0/' + current_file[0: len(current_file) - 1]

    disp_left = Image.open(filename)
    
def generate_disp(imgL, imgR):
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
        disparity = stereo.compute(imgL,imgR)
        return disparity
    
def get_recall(disparity, gt):

    disp = 16
    gt = np.float32(cv2.imread(gt,0))
    gt = np.int16(gt / 255.0 * float(disp))
    disparity = np.int16(np.float32(disparity) / 255.0 * float(disp))
    correct = np.count_nonzero(np.abs(disparity - gt) <= 3)
    return float(correct) / gt.size
    

def iterate_disp():
    perf_list = []
    for i in range(0, len(left_path_list)):
        print("Generating disparity for image:" + str(i))
        print(left_path_list[i])
        print(right_path_list[i])

        imgL = cv2.imread(left_path_list[i],0)
        imgR = cv2.imread(right_path_list[i],0)
        disparity = generate_disp(imgL, imgR)
#        plt.figure()
#        f, axarr = plt.subplots(3,1,gridspec_kw={'hspace':0.75}) 
#        axarr[0].imshow(imgL, "gray")
#        axarr[0].title.set_text('Left Image')
#        axarr[1].imshow(imgR, "gray")
#        axarr[1].title.set_text('Right Image')
#        axarr[2].imshow(disparity, "gray")
#        axarr[2].title.set_text('Disparity Map')
        #plt.show()
        #op_path = "op/img"+str(i)+".png"
        #plt.imsave(op_path, disparity, cmap = 'gray')
        perf = get_recall(disparity, disp_path_list[i]) 
        print("performance -" + str(perf))
        perf_list.append(perf)
    perf_list = np.asarray(perf_list)
    sum_avg = np.mean(perf_list)
    print(sum_avg)
def main():
    data_loader()
    iterate_disp()

if __name__ == '__main__':
    main()