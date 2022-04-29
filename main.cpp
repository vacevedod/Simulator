
// ZED includes
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>
// OpenCV dep
#include <opencv2/cvconfig.h>
//GUI
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
// Sample includes
#include "SaveDepth.h"
#include "dof_gpu.h"

using namespace sl;
cv::Mat slMat2cvMat(Mat& input);

cv::cuda::GpuMat slMat2cvMatGPU(Mat& input);

void printHelp();

int main(int argc, char** argv) {

    // Create a ZED camera object
    Camera zed;

    //helpers
    double norm_depth_focus_point ;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD1080;
    init_params.depth_mode = DEPTH_MODE::ULTRA;
    init_params.coordinate_units = UNIT::MILLIMETER;
    init_params.depth_minimum_distance = 40.0f;
    init_params.depth_maximum_distance = 2000.0f;

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
    Mat gpu_Image_render(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU);
    Mat gpu_transform(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU);
    Mat gpu_depth;
    Mat gpu_depth_normalized;
    Mat gpu_image_convol;

    //trackbar
    char trackbarBlur[7]="DoF";
    char trackbarContrast[9] = "Contrast";
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE); // Create Window
    cv::createTrackbar(trackbarBlur, "Image", 0, 1000);
    cv::createTrackbar(trackbarContrast, "Image", 0, 1000);



    gpu_image_left.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_Image_render.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_transform.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);
    gpu_depth.alloc(camera_resolution_, MAT_TYPE::F32_C1, MEM::GPU);
    gpu_depth_normalized.alloc(camera_resolution_, MAT_TYPE::F32_C1, MEM::GPU);
    gpu_image_convol.alloc(camera_resolution_, MAT_TYPE::U8_C4, MEM::GPU);

    //Calculate gaussian Kernel
    std::vector<float> gauss_vec;

    // Create all the gaussien kernel for different radius and copy them to GPU
    for (int i = 0; i < 32; ++i) {
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
    //testKernel();

    // To share data between sl::Mat and cv::Mat, use slMat2cvMat()
    // Only the headers and pointer to the sl::Mat are copied, not the data itself
    Mat image_zed(new_width, new_height, MAT_TYPE::U8_C4);
    cv::Mat image_ocv = slMat2cvMat(image_zed);

    int kernelRad = 32;

    Mat depth_image_zed_gpu(new_width, new_height, MAT_TYPE::U8_C4, sl::MEM::GPU); // alloc sl::Mat to store GPU depth image
    cv::cuda::GpuMat depth_image_ocv_gpu = slMat2cvMatGPU(depth_image_zed_gpu); // create an opencv GPU reference of the sl::Mat
    cv::Mat depth_image_ocv; // cpu opencv mat for display purposes
    cv::cuda::GpuMat render_gpu = slMat2cvMatGPU(gpu_transform); // create an opencv GPU reference of the sl::Mat
    
    int trackbarPos, trackbarCon;

    // Loop until 'q' is pressed
    char key = ' ';
    while (key != 'q') {

        if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {

            int x = image_zed.getWidth() / 2;
            int y = image_zed.getHeight() / 2;


            // Retrieve Image and Depth
            zed.retrieveImage(gpu_image_left, VIEW::LEFT, MEM::GPU);
            zed.retrieveMeasure(gpu_depth, MEASURE::DEPTH, MEM::GPU);
            //float depth_focus_point = 0.9f;
            float max_range = zed.getInitParameters().depth_maximum_distance;
            float min_range = zed.getInitParameters().depth_minimum_distance;
            //gpu_depth.getValue<sl::float1>(x, y, &depth_focus_point, MEM::GPU);
            // Check that the value is valid
            //if (isValidMeasure(depth_focus_point)) {
            //    std::cout << " Focus point set at : " << depth_focus_point << "mm {" << x << "," << y << "}" << std::endl;
            //    norm_depth_focus_point = (max_range - depth_focus_point) / (max_range - min_range);
            //    norm_depth_focus_point = norm_depth_focus_point > 1.f ? 1.f : (norm_depth_focus_point < 0.f ? 0.f : norm_depth_focus_point);
            //}
            
            trackbarPos = cv::getTrackbarPos(trackbarBlur, "Image");
            
            norm_depth_focus_point = 1.0* trackbarPos/1000.0 ;
            //std::cout << trackbarPos << std::endl;
            //std::cout << norm_depth_focus_point << std::endl;


            // Process Image with CUDA
            // Normalize the depth map and make separable convolution
            normalizeDepth(gpu_depth.getPtr<float>(MEM::GPU), gpu_depth_normalized.getPtr<float>(MEM::GPU), gpu_depth.getStep(MEM::GPU), min_range, max_range, gpu_depth.getWidth(), gpu_depth.getHeight());
            convolutionRows(gpu_image_convol.getPtr<sl::uchar4>(MEM::GPU), gpu_image_left.getPtr<sl::uchar4>(MEM::GPU), gpu_depth_normalized.getPtr<float>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_depth_normalized.getStep(MEM::GPU), norm_depth_focus_point,kernelRad);
            convolutionColumns(gpu_Image_render.getPtr<sl::uchar4>(MEM::GPU), gpu_image_convol.getPtr<sl::uchar4>(MEM::GPU), gpu_depth_normalized.getPtr<float>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_depth_normalized.getStep(MEM::GPU), norm_depth_focus_point,kernelRad);
            
            trackbarCon = cv::getTrackbarPos(trackbarContrast, "Image");
            float p = 1.0 * trackbarCon / 1000.0;
            
            contrast(gpu_Image_render.getPtr<sl::uchar4>(MEM::GPU), gpu_transform.getPtr<sl::uchar4>(MEM::GPU), gpu_image_left.getWidth(), gpu_image_left.getHeight(), gpu_image_left.getStep(MEM::GPU), p);
            
            cv::Mat render_ocv; // cpu opencv mat for display purposes
            

            render_gpu.download(render_ocv);
            cv::cvtColor(render_ocv, render_ocv, cv::COLOR_BGRA2RGBA);
            cv::imshow("Image", render_ocv);
            //cv::imshow("Depth", depth_image_ocv);

            // Handle key event
            key = cv::waitKey(10);
            processKeyEvent(zed, key);
        }
    }


    // sl::Mat GPU memory needs to be free before the zed
    depth_image_zed_gpu.free();

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


cv::Mat slMat2cvMat(Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}


cv::cuda::GpuMat slMat2cvMatGPU(Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::cuda::GpuMat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::GPU), input.getStepBytes(sl::MEM::GPU));
}



void printHelp() {
    std::cout << " Press 's' to save Side by side images" << std::endl;
    std::cout << " Press 'p' to save Point Cloud" << std::endl;
    std::cout << " Press 'd' to save Depth image" << std::endl;
    std::cout << " Press 'm' to switch Point Cloud format" << std::endl;
    std::cout << " Press 'n' to switch Depth format" << std::endl;
}