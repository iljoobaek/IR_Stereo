/*
 *  stereomatch.cpp
 */

#include <opencv4/opencv2/calib3d/calib3d.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/ximgproc.hpp>

#include <stdio.h>

using namespace cv;
using namespace cv::ximgproc;

static void print_help(char** argv)
{
    printf("\nDemo stereo matching converting L and R images into disparity\n");
    printf("\nUsage: %s <left_dir> <right_dir> <out_dir> [-n=<num_imgs>] [--algorithm=sgbm|sgbm3way]\n"
           " [--blocksize=<block_size>] [--max-disparity=<max_disparity>] [-m=<remap_filename>] {no-display||}\n", argv[0]);
}



int main(int argc, char** argv)
{   Mat map11, map12, map21, map22;
    std::string img1_dir = "";
    std::string img2_dir= "";
    std::string op_dir = "";
    std::string img1_filename = "";
    std::string img2_filename = "";
    std::string op_filename = "";
    std::string remap_filename = "";

    enum { STEREO_SGBM=0, STEREO_3WAY=1 };
    int num_imgs, SADWindowSize, numberOfDisparities, alg = STEREO_SGBM;
    bool no_display;
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
    cv::CommandLineParser parser(argc, argv,
        "{@arg1||}{@arg2||}{@arg3||}{help h||}{n|0|}{algorithm||}{blocksize|0|}{max-disparity|0|}{m||}{no-display||}");
    numberOfDisparities = parser.get<int>("max-disparity");
    SADWindowSize = parser.get<int>("blocksize");
    num_imgs = parser.get<int>("n");
    no_display = parser.has("no-display");

    if(parser.has("help"))
    {
        print_help(argv);
        return 0;
    }
    if (parser.has("algorithm"))
    {
        std::string _alg = parser.get<std::string>("algorithm");
        alg = _alg == "sgbm" ? STEREO_SGBM :
            _alg == "sgbm3way" ? STEREO_3WAY : -1;
    }
    if( parser.has("m") ) 
        remap_filename = parser.get<std::string>("m");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    if( alg < 0 )
    {
        printf("Command-line parameter error: Unknown stereo algorithm\n\n");
        print_help(argv);
        return -1;
    }
    if ( numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
    {
        printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
        print_help(argv);
        return -1;
    }
    if (SADWindowSize < 1 || SADWindowSize % 2 != 1)
    {
        printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
        return -1;
    }

    img1_dir = parser.get<std::string>(0);
    img2_dir = parser.get<std::string>(1);
    op_dir = parser.get<std::string>(2);
    if( !remap_filename.empty() )
    {
        // reading intrinsic parameters
        printf("Remapping.. \n");
        FileStorage fs(remap_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", remap_filename.c_str());
            return -1;
        }
        fs["lmapx"] >> map11;
        fs["lmapy"] >> map12;
        fs["rmapx"] >> map21;
        fs["rmapy"] >> map22;
    }

    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setUniquenessRatio(20);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);

    if(alg==STEREO_SGBM)
        sgbm->setMode(StereoSGBM::MODE_SGBM);
    else if(alg==STEREO_3WAY)
        sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);


	for (int i = 1; i <= num_imgs; i++) 
    {   std::string img1_filename = img1_dir + std::to_string(i) + ".tiff";
        std::string img2_filename = img2_dir + std::to_string(i) + ".tiff";
        std::string op_filename = op_dir + std::to_string(i) + ".tiff";

        Mat img1 = imread(img1_filename, 0);
        Mat img2 = imread(img2_filename, 0);
        //bitwise_not(img1,img1);
        //bitwise_not(img2,img2);

        if (img1.empty())
        {
            printf("Command-line parameter error: could not load the first input image file\n");
            return -1;
        }
        if (img2.empty())
        {
            printf("Command-line parameter error: could not load the second input image file\n");
            return -1;
        }

        Size img_size = img1.size();
        numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;
        sgbm->setNumDisparities(numberOfDisparities);
        int cn = img1.channels();
        sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
        sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
        if( !remap_filename.empty() )
        {

        Mat img1r, img2r;
        int64 rect_time = getTickCount();
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);
        rect_time = getTickCount() - rect_time;
        printf("Rectification Time elapsed: %fms\n", rect_time*1000/getTickFrequency());
        img1 = img1r;
        img2 = img2r;
        }

            
        Ptr<DisparityWLSFilter> wls_filter;
        wls_filter = createDisparityWLSFilter(sgbm);
        Ptr<StereoMatcher> right_matcher = createRightMatcher(sgbm);   

        Mat left_disp, right_disp, filtered_disp, disp8;
        int64 disp_time = getTickCount();

        sgbm->compute(img1, img2, left_disp);
        right_matcher->compute(img2, img1, right_disp);

        disp_time = getTickCount() - disp_time;
        printf("Disparity matching Time elapsed: %fms\n", disp_time*1000/getTickFrequency());
        
        wls_filter->setLambda(8000);
        wls_filter->setSigmaColor(1.5);
        int64 filtering_time = getTickCount();
        wls_filter->filter(left_disp,img1,filtered_disp,right_disp);
        filtering_time = (getTickCount() - filtering_time);
        printf("Filtering Time elapsed: %fms\n", filtering_time*1000/getTickFrequency());
        filtered_disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
        imwrite(op_filename, disp8);

        if( !no_display )
        {
            namedWindow("left", 1);
            imshow("left", img1);
            namedWindow("right", 1);
            imshow("right", img2);
            namedWindow("disparity", 0);
            imshow("disparity", disp8);
            printf("press any key to continue...");
            fflush(stdout);
            waitKey();
            printf("\n");
        }
	}
    return 0;
}
