
// ZED includes
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>
// OpenCV dep
#include <opencv2/cvconfig.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>

//GUI
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
// Sample includes
#include "SaveDepth.h"
#include "dof_gpu.h"

#include <chrono>



//UI
#define CVUI_IMPLEMENTATION
#include "cvui.h"



#define WINDOW_NAME	"Low Vision Simulator"

using namespace sl;
//cv::Mat slMat2cvMat(Mat& input);
sl::Mat cvMatGPU2slMat(const cv::cuda::GpuMat& input);
sl::Mat cvMat2slMat(const cv::Mat& input);
cv::cuda::GpuMat slMat2cvMatGPU(Mat& input);
void calculateDFT(cv::Mat& src, cv::Mat& dst);
void fftshift(const cv::Mat& input, cv::Mat& output);
int diopterToGaussRadius(double D);


void printHelp();





int main(int argc, char** argv) {

    // Create a ZED camera object
    Camera zed;

    //helpers
    double norm_depth_focus_point ;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD1080;
    init_params.depth_mode = DEPTH_MODE::NEURAL;
    init_params.coordinate_units = UNIT::MILLIMETER;
    init_params.depth_minimum_distance = 40.0f;
    init_params.depth_maximum_distance = 8000.0f;
    init_params.enable_right_side_measure = true;



    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        printf("%s\n", toString(err).c_str());
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Display help in console
    printHelp();

    // Set runtime parameters after opening the camera
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE::FILL;
    

    // Prepare new image size to retrieve half-resolution images
    Resolution image_size = zed.getCameraInformation().camera_configuration.resolution;


    int new_width = image_size.width ;
    int new_height = image_size.height ;

    Resolution new_image_size(new_width, new_height);
    sl::Resolution camera_resolution_ = zed.getCameraInformation().camera_configuration.resolution;

    Mat gpu_image_left;
    Mat gpu_image_right;
    Mat initialright, initialleft;
    Mat dst1l(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU); 
    Mat dst1r(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU);
    Mat gpu_Image_renderl(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU);
    Mat gpu_Image_renderr(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU);
    Mat gpu_transforml(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU);
    Mat gpu_transformr(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU);
    //Mat gpu_transform2l(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU);
    //Mat gpu_transform2r(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU);
    Mat gpu_depthl;
    Mat gpu_depthr;
    Mat gpu_mask;
    Mat gpu_depthl_normalized;
    Mat gpu_depthr_normalized;

    Mat gpu_image_convoll;
    Mat gpu_image_convolr;

    Mat image32SL;
    Mat image32SR;


    //////////////////////////////////////////////////////////////////////// UI DEFINITION /////////////////////////////////////////
    //values needed for UI
    int width = 400;


    //sigma
    float sigmaXl = 1800, sigmaYl = 1800;
    float sigmaXr = 1800, sigmaYr = 1800;

    //contrast
    double p = 1;
    double yellow = 1;

    //light sensitivity
    int lightSensitivity = 252;
    
    cv::Mat interUI(1500,500,CV_8UC3);

    cv::namedWindow("ImageLeft"); // Create Window
    cv::namedWindow("ImageRight"); // Create Window

    std::cout << new_width << "      " << new_height << std::endl;

    cvui::init(WINDOW_NAME);

    bool fullLowAcuity = false;
    bool tVblack = false;
    bool tVblur=false;
    bool invertMask = false;
    bool blacktunnel = false;
    bool bothEyesControl = false;
    bool sameSigmaXY = false;
    bool temporalglare = false;
    int tunnelVision = 0;

    int fx = 934, fy=484;


    //////////////////////////////////////////////////GLARE INITIALIZATION//////////////////////////////////////////////////
    cv::Mat_<float> a = cv::imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\ZedDepth - Copy\\pupil.png", 0);
    //cv::Mat_<float> a = pupil();

    normalize(a, a, 0, 1, cv::NORM_MINMAX);

    cv::Mat fres = fresnel(0.000575);

    cv::Mat_<float> imgf;
    cv::Mat_<float> fresf;
    fres.convertTo(fresf, CV_32F);
    resize(fresf, fresf, cv::Size(a.rows, a.cols), 0, cv::INTER_LINEAR);


    cv::Mat_<float> complexApertureM(fresf.rows, fresf.cols, CV_32F);;

    //multiply pupil and fresnel term
    complexApertureM = a.mul(fresf);

    //obtain magnitude
    cv::Mat planes1[] = { complexApertureM, cv::Mat::zeros(complexApertureM.size(), CV_32F) };
    cv::Mat compleximg;
    merge(planes1, 2, compleximg);
    dft(compleximg, compleximg);
    cv::Mat real, imaginary;
    cv::Mat planes[] = { real, imaginary };
    split(compleximg, planes);
    cv::Mat mag_img;
    magnitude(planes[0], planes[1], mag_img);
    float lambda = 0.000575;
    int d = 24;
    float K = 1 / (lambda * lambda * d * d);
    mag_img = mag_img * K;
    //mag_img += Scalar::all(1);
    //log(mag_img, mag_img);
    fftshift(mag_img, mag_img);
    normalize(mag_img, mag_img, 0, 1, cv::NORM_MINMAX);


    imshow("Complex aperture", mag_img);
    

    /// number of samples to acquire from human visible spectrum, 32 recommended for efficiency 
    int samples = 32;
    cv::Mat final(1348, 1348, CV_8UC3);
    cv::Mat_<cv::Vec3f> dst2(1348, 1348);

    std::vector<int> values;
    bool colored = true;
    cv::Mat_<float> dstgray(1348, 1348);
    dst2= RGB4Lambda(samples, mag_img,dst2, colored);

    cv::resize(dst2, dst2, cv::Size(512, 512), cv::INTER_NEAREST);
    cv::imshow("bright", dst2);
    cv::cvtColor(dst2, dstgray, cv::COLOR_BGRA2GRAY);

    cv::Mat_<float> colorChannels[3];
    split(dst2, colorChannels);

    cv::cuda::GpuMat d_DFT_psf, d_DFT_psf1, d_DFT_psf2, d_DFT_psf3, d_psf1, d_psf2, d_psf3;
    cv::Mat_<float> h_dftcc[3],h_dftoc;
    bool cudaglare = false;
    int glarechannels = 3;

    if (cudaglare){
        d_psf1.upload(colorChannels[0]);
        d_psf2.upload(colorChannels[1]);
        d_psf3.upload(colorChannels[2]);


        cv::cuda::dft(d_psf1, d_DFT_psf1, cv::Size(colorChannels[0].rows, colorChannels[0].cols), 0);
        cv::cuda::dft(d_psf2, d_DFT_psf2, cv::Size(colorChannels[0].rows, colorChannels[0].cols), 0);
        cv::cuda::dft(d_psf3, d_DFT_psf3, cv::Size(colorChannels[0].rows, colorChannels[0].cols), 0);
    }
    else {
       
        calculateDFT(colorChannels[0], h_dftcc[0]);
        calculateDFT(colorChannels[1], h_dftcc[1]);
        calculateDFT(colorChannels[2], h_dftcc[2]);
        calculateDFT(dstgray, h_dftoc);
        
    }


    
    //////////////////////////////////// MEMORY ALLCATION ZED MATRICES ///////////////////////////////////////////
    
    image32SL.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    image32SR.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    dst1l.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    dst1r.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_image_left.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_image_right.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_Image_renderl.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_Image_renderr.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_transforml.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_transformr.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    //gpu_transform2l.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    //gpu_transform2r.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_depthl.alloc(camera_resolution_, MAT_TYPE::F32_C1, MEM::GPU);
    gpu_depthr.alloc(camera_resolution_, MAT_TYPE::F32_C1, MEM::GPU);
    gpu_mask.alloc(camera_resolution_, MAT_TYPE::U8_C1, MEM::GPU);
    gpu_depthl_normalized.alloc(camera_resolution_, MAT_TYPE::F32_C1, MEM::GPU);
    gpu_depthr_normalized.alloc(camera_resolution_, MAT_TYPE::F32_C1, MEM::GPU);
    gpu_image_convoll.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_image_convolr.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);


    //////////////////////////////////////// GAUSSIAN KERNEL DEPTH OF FIELD  //////////////////////////////////////////////////////////////

    //Calculate gaussian Kernel
    double D = 0;
    //max kernelrad 32, max D is 3.8
    int kernelRadl =32;
    int kernelRadr =32;
    //distortion radius
    int radius = 200;

    double distIntens = 0.2;

    // Create all the gaussien kernel for different radius and copy them to GPU
    createKernel(kernelRadl);
    //testKernel();
    
    //////////////////////////debug ///////////////////////////////////
        
    int intValue1 = 30;
    uchar ucharValue2 = 30;
    char charValue3 = 30;
    float floatValue1 = 12.;
    double doubleValue1 = 15., doubleValue2 = 10.3, doubleValue3 = 2.25;


    ///////////////////////////////////////////////////// ZED MATRICES TO OPENCV MATRICES /////////////////////////////////////////////////////////


    Mat image_zed(new_width, new_height, MAT_TYPE::U8_C4);
    //cv::Mat image_ocv = slMat2cvMat(image_zed);


    Mat depth_image_zed_gpu(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU); // alloc sl::Mat to store GPU depth image
    cv::cuda::GpuMat depth_image_ocv_gpu = slMat2cvMatGPU(depth_image_zed_gpu); // create an opencv GPU reference of the sl::Mat

    Mat depth_image_zed_gpu_r(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU); // alloc sl::Mat to store GPU depth image
    cv::cuda::GpuMat depth_image_ocv_gpu_r = slMat2cvMatGPU(depth_image_zed_gpu_r); // create an opencv GPU reference of the sl::Mat


    cv::Mat depth_image_ocv; // cpu opencv mat for display purposes
    cv::Mat depth_image_ocv_r;
    //cv::cuda::GpuMat render_gpul = slMat2cvMatGPU(gpu_transforml); // create an opencv GPU reference of the sl::Mat
    //cv::cuda::GpuMat render_gpur = slMat2cvMatGPU(gpu_transformr);

    cv::cuda::GpuMat render_gpul = slMat2cvMatGPU(gpu_Image_renderl);
    cv::cuda::GpuMat render_gpur = slMat2cvMatGPU(gpu_Image_renderr);

    Mat slMaskl, slMaskr;
    cv::Mat h_thresholdL, h_thresholdR;
    cv::cuda::GpuMat d_thresholdL = slMat2cvMatGPU(gpu_image_left);
    cv::cuda::GpuMat d_thresholdR = slMat2cvMatGPU(gpu_image_right);


    cv::Mat img32cv;
    cv::Mat img32cvr;
    cv::cuda::GpuMat img32l = slMat2cvMatGPU(image32SL);
    cv::cuda::GpuMat img32r = slMat2cvMatGPU(image32SR);

    cv::cuda::GpuMat testdepth_image_ocv_gpul = slMat2cvMatGPU(slMaskl);
    cv::cuda::GpuMat testdepth_image_ocv_gpur = slMat2cvMatGPU(slMaskr);



    sl::Mat depth;
    cv::cuda::GpuMat d_kernel;

    int top, bottom, left, right;
    // Loop until 'q' is pressed
    int iter = 0;
    /////////////////////////////////////////////////// CAMERA ACQUISITON ///////////////////////////////////////////
    char key = ' ';
    while (key != 'q') {


        if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {
            auto begin = std::chrono::high_resolution_clock::now();
            int x = image_zed.getWidth() / 2;
            int y = image_zed.getHeight() / 2;


            ///////////////////////////////////UI DEFINITION /////////////////////////////////////////////////////////
            //BACKGROUND COLOR 
            interUI = cv::Scalar(50, 50, 50);

            cvui::beginColumn(interUI, 20, 25, -1, -1, 6);

            cvui::text("Low Vis Sim");
            cvui::space(3);


            cvui::text("Type of vision");
            cvui::trackbar(width, &tunnelVision, 0, 3,3);
            cvui::space(5);

            if (tunnelVision == 0)
                cvui::text("20/20 Visual Acuity");
            if (tunnelVision == 1)
                cvui::text("Degraded Visual Acuity");
            if (tunnelVision == 2)
                cvui::text("Tunnel Vision Blur");
            if (tunnelVision == 3)
                cvui::text("Black Tunnel Vision");

            cvui::space(5);

            cvui::text("Contrast");
            cvui::trackbar(width, &p, 1., 0.5, 4);
            cvui::space(5);

            cvui::text("Yellowing");
            cvui::trackbar(width, &yellow, 1., 0.5, 4);
            cvui::space(5);
            
            cvui::checkbox("Tunnel Vision Mask Invert", &invertMask);
            cvui::space(5);

            cvui::checkbox("Temporal Glare", &temporalglare);
            cvui::space(5);

            cvui::checkbox("Color Glare", &colored);
            cvui::space(5);
            (!colored) ? glarechannels = 3 : glarechannels = 1;

            //cvui::text("Kernel Radius Left");
            //cvui::trackbar(width, &kernelRadl, 0, 32);
            //cvui::space(5);
            //
            //cvui::text("Kernel Radius Right");
            //cvui::trackbar(width, &kernelRadr, 0, 32);
            //cvui::space(5);

            cvui::text("Distortion Radius");
            cvui::trackbar(width, &radius, 0, 1000);
            cvui::space(5);

            cvui::text("Distortion Intensity");
            cvui::trackbar(width, &distIntens, -0.2, 0.2, 50);
            cvui::space(5);
\
            cvui::text("General Light Sensitivity");
            cvui::trackbar(width, &lightSensitivity, 0, 255);
            cvui::space(5);

            cvui::text("Depth of Focus");
            cvui::trackbar(width, &norm_depth_focus_point, 0., 1., 4);
            cvui::space(5);

            cvui::text("Diopters");
            cvui::trackbar(width, &D, 0.0, 3.8, 4);
            cvui::space(5);
            kernelRadl = diopterToGaussRadius(D);
            kernelRadr = diopterToGaussRadius(D);

            cvui::text("Sigma X Left Eye");
            cvui::trackbar(width, &sigmaXl, 0.0001f, 2200.f);
            cvui::space(5);

            cvui::text("Sigma Y Left Eye");
            cvui::trackbar(width, &sigmaYl, 0.0001f, 2200.f);
            cvui::space(5);

            cvui::text("Sigma X Right Eye");
            cvui::trackbar(width, &sigmaXr, 0.0001f, 2200.f);
            cvui::space(5);

            cvui::text("Sigma Y Right Eye");
            cvui::trackbar(width, &sigmaYr, 0.0001f, 2200.f);
            cvui::space(5);
            
            cvui::text("Focus Point X");
            cvui::trackbar(width, &fx, 0, 1542);
            cvui::space(5);

            cvui::text("Focus Point Y");
            cvui::trackbar(width, &fy, 0, 870);
            cvui::space(5);

            
            //cvui::text("uchar trackbar, no customization");
            //cvui::trackbar(width, &ucharValue2, (uchar)0, (uchar)255);
            //cvui::space(5);
            
            //cvui::text("signed char trackbar, no customization");
            //cvui::trackbar(width, &charValue3, (char)-128, (char)127);
            //cvui::space(5);


            //cvui::text("double trackbar, label %.1Lf, TRACKBAR_DISCRETE");
            //cvui::trackbar(width, &tunnelVision, 0, 3, 3, "%d", cvui::TRACKBAR_DISCRETE,1);
            //cvui::space(5);

            //cvui::text("double trackbar, label %.2Lf, 2 segments, TRACKBAR_DISCRETE");
            //cvui::trackbar(width, &doubleValue3, 0., 4., 2, "%.2Lf", cvui::TRACKBAR_DISCRETE, 0.25);
            //cvui::space(10);


            cvui::endColumn();



            /////////////////////////////////DYNAMIC GAUSSIAN GENERATION ////////////////////////////////////////////////////
            
            //LEFT
            cv::Rect roi(new_width - fx, new_height - fy, new_width, new_height);
            cv::Mat kernel_Xl = cv::getGaussianKernel(2*new_width, sigmaXl);
            cv::Mat kernel_Yl = cv::getGaussianKernel(2*new_height, sigmaYl);
            cv::Mat kernel_X_transposel;
            cv::transpose(kernel_Xl, kernel_X_transposel);
            cv::Mat kernell = kernel_Yl * kernel_X_transposel;
            cv::Mat subImgl = kernell(roi);
            cv::Mat_<float> mask_vl, proc_imgl;
            normalize(subImgl, mask_vl, 0, 1, cv::NormTypes::NORM_MINMAX);
            
            if (invertMask){
                mask_vl = abs(mask_vl - 1);
                for (int i = 0; i < mask_vl.rows; i++) {
                    for (int j = 0; j < mask_vl.cols; j++) {
                        (mask_vl.at<float>(i, j) < norm_depth_focus_point) ? mask_vl.at<float>(i, j) = 0: mask_vl.at<float>(i, j) = mask_vl.at<float>(i, j) ;
                    }
                }
            }
            //std::cout << mask_vl;
            //cv::imshow("massk", mask_vl);

            //RIGHT
            cv::Mat kernel_Xr = cv::getGaussianKernel(2*new_width, sigmaXr);
            cv::Mat kernel_Yr = cv::getGaussianKernel(2*new_height, sigmaYr);
            cv::Mat kernel_X_transposer;
            cv::transpose(kernel_Xr, kernel_X_transposer);
            cv::Mat kernelr = kernel_Yr * kernel_X_transposer;
            cv::Mat_<float> mask_vr, proc_imgr;
            cv::Mat subImgr = kernelr(roi);
            normalize(subImgr, mask_vr, 0, 1, cv::NormTypes::NORM_MINMAX);
            
            if(invertMask)
                mask_vr = abs(mask_vr - 1);

            //UPLOAD TO CUDA AND SLMAT
            cv::cuda::GpuMat d_maskl;
            d_maskl.upload(mask_vl);
            cvMatGPU2slMat(d_maskl).copyTo(slMaskl, sl::COPY_TYPE::GPU_GPU);

            cv::cuda::GpuMat d_maskr;
            d_maskr.upload(mask_vr);
            cvMatGPU2slMat(d_maskr).copyTo(slMaskr, sl::COPY_TYPE::GPU_GPU);



            //////////////////////////////////////////////////RETRIEVE IMAGE AND DEPTH /////////////////////////////////////////////////////////////

            zed.retrieveImage(gpu_image_left, VIEW::LEFT, MEM::GPU);
            zed.retrieveImage(gpu_image_right, VIEW::RIGHT, MEM::GPU);

            zed.retrieveImage(depth_image_zed_gpu, VIEW::DEPTH, MEM::GPU, new_image_size);
            zed.retrieveImage(depth_image_zed_gpu_r, VIEW::DEPTH_RIGHT, MEM::GPU, new_image_size);

            zed.retrieveMeasure(gpu_depthl, MEASURE::DEPTH, MEM::GPU);
            zed.retrieveMeasure(gpu_depthr, MEASURE::DEPTH_RIGHT, MEM::GPU);

            float max_range = zed.getInitParameters().depth_maximum_distance;
            float min_range = zed.getInitParameters().depth_minimum_distance;
            
           
            // display depth left and right
            //depth_image_ocv_gpu.download(depth_image_ocv);
            //depth_image_ocv_gpu_r.download(depth_image_ocv_r);
            //cv::imshow("Depth", depth_image_ocv);
            //cv::imshow("Depth R", depth_image_ocv_r);
            

            ////////////////////////////////////////////////////////////DFT////////////////////////////////////////////////////////////////////

            gpu_image_left.copyTo(image32SL, sl::COPY_TYPE::GPU_GPU);
            gpu_image_right.copyTo(image32SR, sl::COPY_TYPE::GPU_GPU);
            img32l.download(img32cv);
            img32r.download(img32cvr);
           
            //check exception in memory location h_dftcc
            cv::Mat h_tempdftcc[3] ;
            h_tempdftcc[0] = h_dftcc[0];
            h_tempdftcc[1] = h_dftcc[1];
            h_tempdftcc[2] = h_dftcc[2];
            
            
            //cv::Rect roi2(new_width - fx, new_height - fy, new_width, new_height);
            cv::Mat srcX2(d_thresholdL.rows, d_thresholdL.cols, CV_32F);
            cv::Mat srcY2(d_thresholdL.rows, d_thresholdL.cols, CV_32F);
            cv::Mat srcX(d_thresholdL.rows, d_thresholdL.cols, CV_32F);
            cv::Mat srcY(d_thresholdL.rows, d_thresholdL.cols, CV_32F);
     
            
            cv::cuda::GpuMat afteramdl = distortionMaps(d_thresholdL, fy, fx, radius, srcX, srcY, srcX2, srcY2, distIntens);
            cv::cuda::GpuMat afteramdr = distortionMaps(d_thresholdR, fy, fx, radius, srcX, srcY, srcX2, srcY2, distIntens);
            
            afteramdl.copyTo(d_thresholdL);
            afteramdr.copyTo(d_thresholdR);

            d_thresholdL.download(img32cv);
            d_thresholdR.download(img32cvr);

            //cpu implementation of distortion 
            //cv::Mat afteramdl, afteramdr;
            //
            //afteramdl = distortionMaps(img32cv, fy, fx, radius,  distIntens);
            //afteramdr = distortionMaps(img32cvr, fy, fx, radius, distIntens);
            //img32cv=afteramdl;
            //img32cvr=afteramdr;
            
            if (temporalglare) {
                cv::Mat reschannel[4];
                cv::split(img32cv, reschannel);

                cv::Mat reschannelr[4];
                cv::split(img32cvr, reschannelr);
            
                cv::cuda::GpuMat tmp1, tmp2, tmp3;
                cv::cuda::GpuMat tmp1r, tmp2r, tmp3r;

                cv::cuda::cvtColor(d_thresholdL, tmp1, cv::COLOR_BGRA2GRAY);
                cv::cuda::cvtColor(d_thresholdR, tmp1r, cv::COLOR_BGRA2GRAY);

                cv::cuda::threshold(tmp1, tmp2, lightSensitivity, 255, cv::THRESH_TOZERO);
                cv::cuda::threshold(tmp1r, tmp2r, lightSensitivity, 255, cv::THRESH_TOZERO);

            
                cv::Mat_<float> tp3;
                cv::cuda::normalize(tmp2, tmp3, 0, 1, cv::NORM_MINMAX, -1);
                tmp3.convertTo(tmp3, CV_32F);

                cv::Mat_<float> tp3r;
                cv::cuda::normalize(tmp2r, tmp3r, 0, 1, cv::NORM_MINMAX, -1);
                tmp3r.convertTo(tmp3r, CV_32F);


            
                cv::Mat imgout[3], res[3];
                cv::Mat imgoutr[3], resr[3];

                cv::Mat h_DFT_img, h_DFT_imgr;
                if (cudaglare) {
            
                    cv::cuda::GpuMat d_DFT_img;
                    cv::Mat h_DFT_psf;
                    cv::cuda::GpuMat dres, dres1, dres2, dres3, dimgout, dimgout1, dimgout2, dimgout3;
            
            
                    cv::cuda::resize(tmp3, tmp3, cv::Size(colorChannels[0].rows, colorChannels[0].cols), cv::INTER_LINEAR);
            
                    cv::cuda::dft(tmp3, d_DFT_img, tmp3.size(), 0);
            
                    cv::cuda::multiply(d_DFT_img, d_DFT_psf1, dres1, 1, -1);
                    cv::cuda::multiply(d_DFT_img, d_DFT_psf2, dres2, 1, -1);
                    cv::cuda::multiply(d_DFT_img, d_DFT_psf3, dres3, 1, -1);
            
                    cv::Size dest_size = tmp3.size();
                    int flag = cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE;
                    cv::cuda::dft(dres1, dimgout1, dest_size, flag);
                    cv::cuda::dft(dres2, dimgout2, dest_size, flag);
                    cv::cuda::dft(dres3, dimgout3, dest_size, flag);
            
            
                    dimgout1.download(imgout[0]);
                    dimgout2.download(imgout[1]);
                    dimgout3.download(imgout[2]);
            
                }
                else {
                    tmp3.download(h_thresholdL);
                    tmp3r.download(h_thresholdR);

                    
            
                    h_thresholdL.convertTo(h_thresholdL, CV_32F);
                    resize(h_thresholdL, h_thresholdL, h_dftcc[1].size(), cv::INTER_NEAREST);
                    calculateDFT(h_thresholdL, h_DFT_img);

                    h_thresholdR.convertTo(h_thresholdR, CV_32F);
                    resize(h_thresholdR, h_thresholdR, h_dftcc[1].size(), cv::INTER_NEAREST);
                    calculateDFT(h_thresholdR, h_DFT_imgr);
                    
            
                    //cv::imshow("temp", h_thresholdR);
                    if (colored) {
                        spectMultiplication(h_DFT_img, h_dftcc, res, imgout, new_width, new_height, reschannel);
                        spectMultiplication(h_DFT_imgr, h_dftcc, resr, imgoutr, new_width, new_height, reschannelr);
            
                    }
                    else {
                        cv::mulSpectrums(h_DFT_img, h_dftoc, res[0], 0);
                        dft(res[0], imgout[0], cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
                        fftshift(imgout[0], imgout[0]);
                        normalize(imgout[0], imgout[0], 0, 255, cv::NORM_MINMAX);
                        imgout[0].convertTo(imgout[0], CV_8UC1);
                        cv::resize(imgout[0], imgout[0], cv::Size(new_width, new_height), cv::INTER_LINEAR);
                        reschannel[0] = reschannel[0] + imgout[0];
                        reschannel[1] = reschannel[1] + imgout[0];
                        reschannel[2] = reschannel[2] + imgout[0];

                        cv::mulSpectrums(h_DFT_imgr, h_dftoc, resr[0], 0);
                        dft(resr[0], imgoutr[0], cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
                        fftshift(imgoutr[0], imgoutr[0]);
                        normalize(imgoutr[0], imgoutr[0], 0, 255, cv::NORM_MINMAX);
                        imgoutr[0].convertTo(imgoutr[0], CV_8UC1);
                        cv::resize(imgoutr[0], imgoutr[0], cv::Size(new_width, new_height), cv::INTER_LINEAR);
                        reschannelr[0] = reschannelr[0] + imgoutr[0];
                        reschannelr[1] = reschannelr[1] + imgoutr[0];
                        reschannelr[2] = reschannelr[2] + imgoutr[0];
                    }
            
            
            
                    reschannel[3] = 255;
                    reschannelr[3] = 255;
                    
                    std::vector<cv::Mat> RESchannels{ reschannel[0], reschannel[1], reschannel[2], reschannel[3] };
                    std::vector<cv::Mat> RESchannelsr{ reschannelr[0], reschannelr[1], reschannelr[2], reschannelr[3] };
                    
                    merge(RESchannels, img32cv);
                    merge(RESchannelsr, img32cvr);

                    d_thresholdL.upload(img32cv);
                    d_thresholdR.upload(img32cvr);

                }
            }

            /////////////////////////////////IMAGE PROCESSION CUDA FUNCTIONS ////////////////////////////////////////////////////////////////////


            
            cvMatGPU2slMat(d_thresholdL).copyTo(initialleft, sl::COPY_TYPE::GPU_GPU);
            cvMatGPU2slMat(d_thresholdR).copyTo(initialright, sl::COPY_TYPE::GPU_GPU);



            normalizeDepth(gpu_depthl.getPtr<float>(MEM::GPU), gpu_depthl_normalized.getPtr<float>(MEM::GPU), gpu_depthl.getStep(MEM::GPU), min_range, max_range, gpu_depthl.getWidth(), gpu_depthl.getHeight());
            normalizeDepth(gpu_depthr.getPtr<float>(MEM::GPU), gpu_depthr_normalized.getPtr<float>(MEM::GPU), gpu_depthr.getStep(MEM::GPU), min_range, max_range, gpu_depthr.getWidth(), gpu_depthr.getHeight());


            colorShift(initialleft.getPtr<sl::uchar4>(MEM::GPU), dst1l.getPtr<sl::uchar4>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_image_left.getStep(MEM::GPU), yellow);
            colorShift(initialright.getPtr<sl::uchar4>(MEM::GPU), dst1r.getPtr<sl::uchar4>(MEM::GPU), gpu_image_right.getWidth(), gpu_image_right.getHeight(), gpu_image_right.getStep(MEM::GPU), yellow);

            contrast(dst1l.getPtr<sl::uchar4>(MEM::GPU), gpu_transforml.getPtr<sl::uchar4>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_image_left.getStep(MEM::GPU), p);
            contrast(dst1r.getPtr<sl::uchar4>(MEM::GPU), gpu_transformr.getPtr<sl::uchar4>(MEM::GPU), gpu_image_right.getWidth(), gpu_image_right.getHeight(), gpu_image_right.getStep(MEM::GPU), p);


            convolutionRows(gpu_image_convoll.getPtr<sl::uchar4>(MEM::GPU), gpu_transforml.getPtr<sl::uchar4>(MEM::GPU), gpu_depthl_normalized.getPtr<float>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_depthl_normalized.getStep(MEM::GPU), norm_depth_focus_point, kernelRadl, tunnelVision, slMaskl.getPtr<float>(MEM::GPU), slMaskl.getStep(MEM::GPU));
            convolutionRows(gpu_image_convolr.getPtr<sl::uchar4>(MEM::GPU), gpu_transformr.getPtr<sl::uchar4>(MEM::GPU), gpu_depthr_normalized.getPtr<float>(MEM::GPU), gpu_image_right.getWidth(), gpu_image_right.getHeight(), gpu_depthr_normalized.getStep(MEM::GPU), norm_depth_focus_point, kernelRadr, tunnelVision, slMaskr.getPtr<float>(MEM::GPU), slMaskr.getStep(MEM::GPU));

            convolutionColumns(gpu_Image_renderl.getPtr<sl::uchar4>(MEM::GPU), gpu_image_convoll.getPtr<sl::uchar4>(MEM::GPU), gpu_depthl_normalized.getPtr<float>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_depthl_normalized.getStep(MEM::GPU), norm_depth_focus_point, kernelRadl, tunnelVision, slMaskl.getPtr<float>(MEM::GPU), slMaskl.getStep(MEM::GPU));
            convolutionColumns(gpu_Image_renderr.getPtr<sl::uchar4>(MEM::GPU), gpu_image_convolr.getPtr<sl::uchar4>(MEM::GPU), gpu_depthr_normalized.getPtr<float>(MEM::GPU), gpu_image_right.getWidth(), gpu_image_right.getHeight(), gpu_depthr_normalized.getStep(MEM::GPU), norm_depth_focus_point, kernelRadr, tunnelVision, slMaskr.getPtr<float>(MEM::GPU), slMaskr.getStep(MEM::GPU));


            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - begin;
            std::cout << "Total time:" << diff.count() << std::endl;
                    


            //////////////////////////////////////////////////////DOWNLOAD FOR RENDER //////////////////////////////////////////////////////
            cv::Mat render_ocvl, render_ocvr; // cpu opencv mat for display purposes
            cv::Mat gpumaskdownloadl, gpumaskdownloadr;
            
            render_gpul.download(gpumaskdownloadl);
            render_gpur.download(gpumaskdownloadr);

            //////////////////////////////////////////////CORRECT OPENCV COLORS //////////////////////////////////////////////////////////////////////

            //cv::cvtColor(render_ocvl, render_ocvl, cv::COLOR_BGRA2RGBA);
            //cv::cvtColor(render_ocvr, render_ocvr, cv::COLOR_BGRA2RGBA);

            //cv::cvtColor(gpumaskdownloadl, gpumaskdownloadl, cv::COLOR_BGRA2RGBA);
            //cv::cvtColor(gpumaskdownloadr, gpumaskdownloadr, cv::COLOR_BGRA2RGBA);


           //////////////////////////////////////////////////////DISPLAY  ////////////////////////////////////////////////////////////////////////////////

            cv::imshow("ImageLeft", gpumaskdownloadl);
            cv::imshow("ImageRight", gpumaskdownloadr);
            

            cvui::update(WINDOW_NAME);
            cvui::imshow(WINDOW_NAME, interUI);

            // Handle key event
            key = cv::waitKey(10);
            processKeyEvent(zed, key);
        }
    }


    ////////////////////////////////////////////////////////////FREE MEMORY//////////////////////////////////////////////////////////////////////////////
    depth_image_zed_gpu.free();
    gpu_depthl.free();
    gpu_depthl_normalized.free();
    
    gpu_depthr.free();
    gpu_depthr_normalized.free();
    gpu_image_convoll.free();
    gpu_image_left.free();
    //gpu_Image_render.free();
    gpu_transforml.free();
    gpu_transformr.free();

    //gpu_transform2l.free();
    //gpu_transform2r.free();
    zed.close();
    return 0;
}



// Mapping between MAT_TYPE and CV_TYPE
int getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    switch (type) {
    case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
    case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
    case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
    case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
    case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
    case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
    case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
    case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
    default: break;
    }
    return cv_type;
}

sl::Mat cvMat2slMat(const cv::Mat& input) {
    sl::MAT_TYPE sl_type;
    switch (input.type()) {
    case CV_32FC1: sl_type = sl::MAT_TYPE::F32_C1;
        break;
    case CV_32FC2: sl_type = sl::MAT_TYPE::F32_C2;
        break;
    case CV_32FC3: sl_type = sl::MAT_TYPE::F32_C3;
        break;
    case CV_32FC4: sl_type = sl::MAT_TYPE::F32_C4;
        break;
    case CV_8UC1: sl_type = sl::MAT_TYPE::U8_C1;
        break;
    case CV_8UC2: sl_type = sl::MAT_TYPE::U8_C2;
        break;
    case CV_8UC3: sl_type = sl::MAT_TYPE::U8_C3;
        break;
    case CV_8UC4: sl_type = sl::MAT_TYPE::U8_C4;
        break;
    default: break;
    }
    return sl::Mat(input.cols, input.rows, sl::MAT_TYPE::F32_C1, input.data, input.step, sl::MEM::CPU);
}

cv::cuda::GpuMat slMat2cvMatGPU(Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::cuda::GpuMat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::GPU), input.getStepBytes(sl::MEM::GPU));
}

sl::Mat cvMatGPU2slMat(const cv::cuda::GpuMat& input) {
    
    return sl::Mat(input.cols, input.rows, sl::MAT_TYPE::F32_C1, input.data, input.step, sl::MEM::GPU);
}

void printHelp() {
    std::cout << " Press 's' to save Side by side images" << std::endl;
    std::cout << " Press 'p' to save Point Cloud" << std::endl;
    std::cout << " Press 'd' to save Depth image" << std::endl;
    std::cout << " Press 'm' to switch Point Cloud format" << std::endl;
    std::cout << " Press 'n' to switch Depth format" << std::endl;
}

int diopterToGaussRadius(double D) {
    //float D = 3.8;
    //b is blur disk in degrees of visual angle
    float pupilDiameter = 3.34;
    float b = 0.057 * (pupilDiameter)*D;
    
    //specific for dell u4919dw
    float degreeToPixel = b * 100 / 2.23;
    return (int)degreeToPixel;

}








































































































































































































































































































































































































