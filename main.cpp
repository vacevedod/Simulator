
// ZED includes
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>
// OpenCV dep
#include <opencv2/cvconfig.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/cudaarithm.hpp>

//GUI
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
// Sample includes
#include "SaveDepth.h"
#include "dof_gpu.h"


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

void printHelp();
std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "this8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}




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
    init_params.depth_maximum_distance = 2000.0f;
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

    //////////////////////////////////////////////////////////////////////// UI DEFINITION /////////////////////////////////////////
    //values needed for UI
    int width = 400;


    //sigma
    float sigmaXl = 1800, sigmaYl = 1800;
    float sigmaXr = 1800, sigmaYr = 1800;

    //contrast
    double p = 1;
    
    cv::Mat interUI(1000,500,CV_8UC3);

    cv::namedWindow("ImageLeft"); // Create Window
    cv::namedWindow("ImageRight"); // Create Window



    cvui::init(WINDOW_NAME);

    bool fullLowAcuity = false;
    bool tVblack = false;
    bool tVblur=false;
    bool invertMask = false;
    bool blacktunnel = false;
    bool bothEyesControl = false;
    bool sameSigmaXY = false;
    int tunnelVision = 0;

    cv::Mat psf = cv::imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\ZedDepth - Copy\\SpectralPSF2.png", 0);
    psf.convertTo(psf, CV_32F);



    //////////////////////////////////// MEMORY ALLCATION ZED MATRICES ///////////////////////////////////////////
    
    image32SL.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
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
    std::vector<float> gauss_vec;
    int kernelRadl = 32;
    int kernelRadr = 32;

    // Create all the gaussien kernel for different radius and copy them to GPU
    for (int i = 0; i < kernelRadl; ++i) {
        gauss_vec.resize((i + 1) * 2 + 1, 0);

        // Compute Gaussian coeff
        int rad = (gauss_vec.size() - 1) / 2;
        float ssigma = 0.3f * ((gauss_vec.size() - 1.f) * 0.5f - 1.f) + 0.8f;
        float sum = 0;
        for (int u = -rad; u <= rad; u++) {
            float gauss_value = expf(-1.f * (powf(u, 2.f) / (2.f * powf(ssigma, 2.f))));
            gauss_vec[u + rad] = gauss_value;
            sum += gauss_value;
        }
        sum = 1.f / sum;
        for (int u = 0; u < gauss_vec.size(); u++)
            gauss_vec[u] *= sum;
        
        // Copy coeff to GPU
        copyKernel(gauss_vec.data(), i);
    }


    
    //////////////////////////debug ///////////////////////////////////
        //testKernel();
    int intValue1 = 30;
    uchar ucharValue2 = 30;
    char charValue3 = 30;
    float floatValue1 = 12.;
    double doubleValue1 = 15., doubleValue2 = 10.3, doubleValue3 = 2.25;


 

    ///////////////////////////////////////////////////// ZED MATRICES TO OPENCV MATRICES /////////////////////////////////////////////////////////

    // To share data between sl::Mat and cv::Mat, use slMat2cvMat()
    // Only the headers and pointer to the sl::Mat are copied, not the data itself
    Mat image_zed(new_width, new_height, MAT_TYPE::U8_C4);
    //cv::Mat image_ocv = slMat2cvMat(image_zed);


    Mat depth_image_zed_gpu(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU); // alloc sl::Mat to store GPU depth image
    cv::cuda::GpuMat depth_image_ocv_gpu = slMat2cvMatGPU(depth_image_zed_gpu); // create an opencv GPU reference of the sl::Mat
    cv::Mat depth_image_ocv; // cpu opencv mat for display purposes
    //cv::cuda::GpuMat render_gpul = slMat2cvMatGPU(gpu_transforml); // create an opencv GPU reference of the sl::Mat
    //cv::cuda::GpuMat render_gpur = slMat2cvMatGPU(gpu_transformr);

    cv::cuda::GpuMat render_gpul = slMat2cvMatGPU(gpu_transforml);
    cv::cuda::GpuMat render_gpur = slMat2cvMatGPU(gpu_transformr);

    Mat slMaskl, slMaskr;
    cv::Mat h_thresholdL;
    cv::cuda::GpuMat d_thresholdL = slMat2cvMatGPU(gpu_image_left);


    cv::Mat img32cv;
    cv::cuda::GpuMat img32 = slMat2cvMatGPU(image32SL);

    cv::cuda::GpuMat testdepth_image_ocv_gpul = slMat2cvMatGPU(slMaskl);
    cv::cuda::GpuMat testdepth_image_ocv_gpur = slMat2cvMatGPU(slMaskr);



    sl::Mat depth;
    cv::cuda::GpuMat d_kernel;

    int top, bottom, left, right;
    // Loop until 'q' is pressed

    /////////////////////////////////////////////////// CAMERA ACQUISITON ///////////////////////////////////////////
    char key = ' ';
    while (key != 'q') {


        if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {

            int x = image_zed.getWidth() / 2;
            int y = image_zed.getHeight() / 2;

            ///////////////////////////////////UI DEFINITION /////////////////////////////////////////////////////////
            //BACKGROUND COLOR 
            interUI = cv::Scalar(50, 50, 50);

            cvui::beginColumn(interUI, 20, 25, -1, -1, 6);

            cvui::text("Low Vis Sim");
            cvui::space(3);

            cvui::text("Contrast");
            cvui::trackbar(width, &p, 1., 0.5, 4);
            cvui::space(5);

            cvui::checkbox("Low Visual Acuity", &fullLowAcuity);
            cvui::space(5);

            cvui::checkbox("Tunnel Vision Black", &tVblur);
            cvui::space(5);
            if (tVblack) {
                tVblur = false;
                tunnelVision = 2;
            }

            cvui::checkbox("Tunnel Vision Blur", &tVblack);
            cvui::space(5);
            if (tVblur) {
                tVblack = false;
                tunnelVision = 3;
            }

            cvui::checkbox("Tunnel Vision Mask Invert", &invertMask);
            cvui::space(5);

            cvui::text("Kernel Radius");
            cvui::trackbar(width, &kernelRadl, 0, 32);
            cvui::space(5);

            cvui::text("Kernel Redius");
            cvui::trackbar(width, &kernelRadr, 0, 32);
            cvui::space(5);


            cvui::text("Depth of Focus");
            cvui::trackbar(width, &norm_depth_focus_point, 0., 1., 4);
            cvui::space(5);

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

            //cvui::text("uchar trackbar, no customization");
            //cvui::trackbar(width, &ucharValue2, (uchar)0, (uchar)255);
            //cvui::space(5);
            
            //cvui::text("signed char trackbar, no customization");
            //cvui::trackbar(width, &charValue3, (char)-128, (char)127);
            //cvui::space(5);


            //cvui::text("double trackbar, label %.1Lf, TRACKBAR_DISCRETE");
            //cvui::trackbar(width, &doubleValue2, 10., 10.5, 1, "%.1Lf", cvui::TRACKBAR_DISCRETE, 0.1);
            //cvui::space(5);

            //cvui::text("double trackbar, label %.2Lf, 2 segments, TRACKBAR_DISCRETE");
            //cvui::trackbar(width, &doubleValue3, 0., 4., 2, "%.2Lf", cvui::TRACKBAR_DISCRETE, 0.25);
            //cvui::space(10);


            cvui::endColumn();



            /////////////////////////////////DYNAMIC GAUSSIAN GENERATION ////////////////////////////////////////////////////
            
            //LEFT
            cv::Mat kernel_Xl = cv::getGaussianKernel(new_width, sigmaXl);
            cv::Mat kernel_Yl = cv::getGaussianKernel(new_height, sigmaYl);
            cv::Mat kernel_X_transposel;
            cv::transpose(kernel_Xl, kernel_X_transposel);
            cv::Mat kernell = kernel_Yl * kernel_X_transposel;
            cv::Mat_<float> mask_vl, proc_imgl;
            normalize(kernell, mask_vl, 0, 1, cv::NormTypes::NORM_MINMAX);

            if (invertMask)
                mask_vl = abs(mask_vl - 1);

            //RIGHT
            cv::Mat kernel_Xr = cv::getGaussianKernel(new_width, sigmaXr);
            cv::Mat kernel_Yr = cv::getGaussianKernel(new_height, sigmaYr);
            cv::Mat kernel_X_transposer;
            cv::transpose(kernel_Xr, kernel_X_transposer);
            cv::Mat kernelr = kernel_Yr * kernel_X_transposer;
            cv::Mat_<float> mask_vr, proc_imgr;
            //mask_vr = kernelr;
            normalize(kernelr, mask_vr, 0, 1, cv::NormTypes::NORM_MINMAX);
            
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

            zed.retrieveMeasure(gpu_depthl, MEASURE::DEPTH, MEM::GPU);
            zed.retrieveMeasure(gpu_depthr, MEASURE::DEPTH_RIGHT, MEM::GPU);

            float max_range = zed.getInitParameters().depth_maximum_distance;
            float min_range = zed.getInitParameters().depth_minimum_distance;
            
            imshow("maks", mask_vl);
            depth_image_ocv_gpu.download(depth_image_ocv);


            cv::imshow("Depth", depth_image_ocv);

            /////////////////////////////////IMAGE PROCESSION CUDA FUNCTIONS ////////////////////////////////////////////////////////////////////
            

            

            normalizeDepth(gpu_depthl.getPtr<float>(MEM::GPU), gpu_depthl_normalized.getPtr<float>(MEM::GPU), gpu_depthl.getStep(MEM::GPU), min_range, max_range, gpu_depthl.getWidth(), gpu_depthl.getHeight());
            normalizeDepth(gpu_depthr.getPtr<float>(MEM::GPU), gpu_depthr_normalized.getPtr<float>(MEM::GPU), gpu_depthr.getStep(MEM::GPU), min_range, max_range, gpu_depthr.getWidth(), gpu_depthr.getHeight());

            convolutionRows(gpu_image_convoll.getPtr<sl::uchar4>(MEM::GPU), gpu_image_left.getPtr<sl::uchar4>(MEM::GPU), gpu_depthl_normalized.getPtr<float>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_depthl_normalized.getStep(MEM::GPU), norm_depth_focus_point,kernelRadl, fullLowAcuity, slMaskl.getPtr<float>(MEM::GPU), slMaskl.getStep(MEM::GPU));
            convolutionRows(gpu_image_convolr.getPtr<sl::uchar4>(MEM::GPU), gpu_image_right.getPtr<sl::uchar4>(MEM::GPU), gpu_depthr_normalized.getPtr<float>(MEM::GPU), gpu_image_right.getWidth(), gpu_image_right.getHeight(), gpu_depthr_normalized.getStep(MEM::GPU), norm_depth_focus_point, kernelRadr, fullLowAcuity, slMaskr.getPtr<float>(MEM::GPU), slMaskr.getStep(MEM::GPU));
            
            convolutionColumns(gpu_Image_renderl.getPtr<sl::uchar4>(MEM::GPU), gpu_image_convoll.getPtr<sl::uchar4>(MEM::GPU), gpu_depthl_normalized.getPtr<float>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_depthl_normalized.getStep(MEM::GPU), norm_depth_focus_point,kernelRadl, fullLowAcuity, slMaskl.getPtr<float>(MEM::GPU), slMaskl.getStep(MEM::GPU));
            convolutionColumns(gpu_Image_renderr.getPtr<sl::uchar4>(MEM::GPU), gpu_image_convolr.getPtr<sl::uchar4>(MEM::GPU), gpu_depthr_normalized.getPtr<float>(MEM::GPU), gpu_image_right.getWidth(), gpu_image_right.getHeight(), gpu_depthr_normalized.getStep(MEM::GPU), norm_depth_focus_point, kernelRadr, fullLowAcuity, slMaskr.getPtr<float>(MEM::GPU), slMaskr.getStep(MEM::GPU));

          

            contrast(gpu_Image_renderl.getPtr<sl::uchar4>(MEM::GPU), gpu_transforml.getPtr<sl::uchar4>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_image_left.getStep(MEM::GPU), p, slMaskl.getPtr<float>(MEM::GPU), slMaskl.getStep(MEM::GPU));
            contrast(gpu_Image_renderr.getPtr<sl::uchar4>(MEM::GPU), gpu_transformr.getPtr<sl::uchar4>(MEM::GPU), gpu_image_right.getWidth(), gpu_image_right.getHeight(), gpu_image_right.getStep(MEM::GPU), p, slMaskr.getPtr<float>(MEM::GPU), slMaskr.getStep(MEM::GPU));

            
            ////////////////////////////////////////////////////////////DFT////////////////////////////////////////////////////////////////////
            gpu_image_left.copyTo(image32SL, sl::COPY_TYPE::GPU_GPU);
            img32.download(img32cv);
            cv::cuda::threshold(d_thresholdL, d_thresholdL, 252, 255, cv::THRESH_TOZERO);
            d_thresholdL.download(h_thresholdL);
            cv::cvtColor(h_thresholdL, h_thresholdL, cv::COLOR_BGRA2GRAY);
            //cv::imshow("wite", h_thresholdL);
            
            cv::normalize(h_thresholdL, h_thresholdL, 0, 1, cv::NORM_MINMAX);
            
            h_thresholdL.convertTo(h_thresholdL, CV_32F);
            cv::cuda::GpuMat d_DFT_img, d_DFT_psf;
            cv::Mat h_DFT_img, h_DFT_psf;
            
            cv::Mat res, imgout;
            top = (int)(0.25 * h_thresholdL.rows); bottom = top;
            left = (int)(0.25 * h_thresholdL.cols); right = left;
            
            //cv::copyMakeBorder(h_thresholdL, h_thresholdL, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::resize(h_thresholdL, h_thresholdL, cv::Size(psf.rows, psf.cols), cv::INTER_LINEAR);
            calculateDFT(h_thresholdL, h_DFT_img);
            
            calculateDFT(psf, h_DFT_psf);
            cv::mulSpectrums(h_DFT_img, h_DFT_psf, res, 0);
            dft(res, imgout, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
            fftshift(imgout, imgout);
            normalize(imgout, imgout, 0, 255, cv::NORM_MINMAX);
            
            imgout.convertTo(imgout,CV_8UC1);
            
            
            cv::resize(imgout, imgout, cv::Size(new_width, new_height), cv::INTER_LINEAR);
            
            
            cv::Mat out;
            cv::Mat alpha (imgout.rows, imgout.cols, CV_8UC1, cv::Scalar(255));
            cv::Mat in[] = { imgout, imgout, imgout, alpha};
            cv::merge(in, 4, out);
            
            out.convertTo(out, CV_8UC4);
            //img32cv.convertTo(img32cv, CV_8UC);
            img32cv = img32cv + out;
            
            //h_thresholdL.convertTo(h_thresholdL, CV_8UC1);
            //imgout.convertTo(imgout, CV_8UC1);
            //cv::Mat out;
            //
            //out = h_thresholdL + imgout;
            //
            //normalize(imgout, imgout, 0, 255, cv::NORM_MINMAX);
            
            //cv::imshow("left threshold", img32cv);
            ///////////////////////////////////////////////////////////CONTRAST A TUNNEL VISION////////////////////////////////////////////////////////////////////

            //////////////////////////////////////////////////////DOWNLOAD FOR RENDER //////////////////////////////////////////////////////
            cv::Mat render_ocvl, render_ocvr; // cpu opencv mat for display purposes
            cv::Mat gpumaskdownloadl, gpumaskdownloadr;
            
            //render_gpul.download(render_ocvl);
            //render_gpur.download(render_ocvr);
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

void calculateDFT(cv::Mat& src, cv::Mat& dst) {
    cv::Mat planes[] = { src, cv::Mat::zeros(src.size(), CV_32F) };
    cv::Mat compleximg;
    cv::merge(planes, 2, compleximg);
    cv::dft(compleximg, compleximg);
    dst = compleximg;
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

void fftshift(const cv::Mat& input, cv::Mat& output) {
    output = input.clone();
    int cx = output.cols / 2;
    int cy = output.rows / 2;
    cv::Mat q1(output, cv::Rect(0, 0, cx, cy));
    cv::Mat q2(output, cv::Rect(cx, 0, cx, cy));
    cv::Mat q3(output, cv::Rect(0, cy, cx, cy));
    cv::Mat q4(output, cv::Rect(cx, cy, cx, cy));

    cv::Mat temp;
    q1.copyTo(temp);
    q4.copyTo(q1);
    temp.copyTo(q4);
    q2.copyTo(temp);
    q3.copyTo(q2);
    temp.copyTo(q3);


}











































































































































































































































































































































































































