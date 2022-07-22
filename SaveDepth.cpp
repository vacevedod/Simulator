#include "SaveDepth.h"
#include "dof_gpu.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>


using namespace sl;
using namespace std;

int count_save = 0;
int mode_PointCloud = 0;
int mode_Depth = 0;
int PointCloud_format;
int Depth_format;

std::string PointCloud_format_ext = ".ply";
std::string Depth_format_ext = ".png";

#define PI 3.14159265f

void setPointCloudFormatName(int format) {
    switch (format) {
    case 0:
        PointCloud_format_ext = ".xyz";
        break;
    case  1:
        PointCloud_format_ext = ".pcd";
        break;
    case  2:
        PointCloud_format_ext = ".ply";
        break;
    case  3:
        PointCloud_format_ext = ".vtk";
        break;
    default:
        break;
    }
}

void setDepthFormatName(int format) {
    switch (format) {
    case  0:
        Depth_format_ext = ".png";
        break;
    case  1:
        Depth_format_ext = ".pfm";
        break;
    case  2:
        Depth_format_ext = ".pgm";
        break;
    default:
        break;
    }
}

void processKeyEvent(Camera& zed, char& key) {
    switch (key) {
    case 'd':
    case 'D':
        saveDepth(zed, path + prefixDepth + to_string(count_save));
        break;

    case 'n': // Depth format
    case 'N':
    {
        mode_Depth++;
        Depth_format = (mode_Depth % 3);
        setDepthFormatName(Depth_format);
        std::cout << "Depth format: " << Depth_format_ext << std::endl;
    }
    break;

    case 'p':
    case 'P':
        savePointCloud(zed, path + prefixPointCloud + to_string(count_save));
        break;


    case 'm': // Point cloud format
    case 'M':
    {
        mode_PointCloud++;
        PointCloud_format = (mode_PointCloud % 4);
        setPointCloudFormatName(PointCloud_format);
        std::cout << "Point Cloud format: " << PointCloud_format_ext << std::endl;
    }
    break;

    case 'h': // Print help
    case 'H':
        cout << helpString << endl;
        break;

    case 's': // Save side by side image
    case 'S':
        saveSbSImage(zed, std::string("ZED_image") + std::to_string(count_save) + std::string(".png"));
        break;
    }
    count_save++;
}

void savePointCloud(Camera& zed, std::string filename) {
    std::cout << "Saving Point Cloud... " << flush;

    sl::Mat point_cloud;
    zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA);

    auto state = point_cloud.write((filename + PointCloud_format_ext).c_str());

    if (state == ERROR_CODE::SUCCESS)
        std::cout << "Point Cloud has been saved under " << filename << PointCloud_format_ext << endl;
    else
        std::cout << "Failed to save point cloud... Please check that you have permissions to write at this location (" << filename << "). Re-run the sample with administrator rights under windows" << endl;
}

void saveDepth(Camera& zed, std::string filename) {
    std::cout << "Saving Depth Map... " << flush;

    sl::Mat depth;
    zed.retrieveMeasure(depth, sl::MEASURE::DEPTH);

    convertUnit(depth, zed.getInitParameters().coordinate_units, UNIT::MILLIMETER);
    auto state = depth.write((filename + Depth_format_ext).c_str());

    if (state == ERROR_CODE::SUCCESS)
        std::cout << "Depth Map has been save under " << filename << Depth_format_ext << endl;
    else
        std::cout << "Failed to save depth map... Please check that you have permissions to write at this location (" << filename << "). Re-run the sample with administrator rights under windows" << endl;
}

void saveSbSImage(Camera& zed, std::string filename) {
    sl::Mat image_sbs;
    zed.retrieveImage(image_sbs, sl::VIEW::SIDE_BY_SIDE);

    auto state = image_sbs.write(filename.c_str());

    if (state == sl::ERROR_CODE::SUCCESS)
        std::cout << "Side by Side image has been save under " << filename << endl;
    else
        std::cout << "Failed to save image... Please check that you have permissions to write at this location (" << filename << "). Re-run the sample with administrator rights under windows" << endl;
}


float lerp(float a, float b, float t) { return a + t * (b - a); }


float interpolate_pix(float pix1, float pix2, float pix3,
    float pix4, float intpX, float intpY, cv::Vec3f bgr) {
    // First horizontal interpolation
    float B_h1 = lerp(pix1, pix2, intpX);
    float G_h1 = lerp(pix1, pix2, intpX);
    float R_h1 = lerp(pix1, pix2, intpX);

    // Second horizontal interpolation
    float B_h2 = lerp(pix3, pix4, intpX);
    float G_h2 = lerp(pix3, pix4, intpX);
    float R_h2 = lerp(pix3, pix4, intpX);

    // Vertical interpolation
    float B = lerp(B_h1, B_h2, intpY);
    float G = lerp(G_h1, G_h2, intpY);
    float R = lerp(R_h1, R_h2, intpY);

    //B = B*bgr[0];
    //G = G*bgr[1];
    //R = R*bgr[2];
    //
    //R >= 0 ? R = R : R = 0;
    //G >= 0 ? G = G : G = 0;
    //B >= 0 ? B = B : B = 0;
    //
    //R <= 255 ? R = R : R = 255;
    //G <= 255 ? G = G : G = 255;
    //B <= 255 ? B = B : B = 255;

    //return cv::Vec3f(B,G,R);
    return B;
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

void calculateDFT(cv::Mat& src, cv::Mat& dst) {
    cv::Mat planes[] = { src, cv::Mat::zeros(src.size(), CV_32F) };
    cv::Mat compleximg;
    cv::merge(planes, 2, compleximg);
    cv::dft(compleximg, compleximg);
    dst = compleximg;
}

cv::Mat fresnel(float lambda) {
    cv::Mat_<float> fresn(1024, 1024, CV_8UC1);
    int width = fresn.rows;
    int height = fresn.cols;
    float resolution = height / 9.0;
    int d = 20;

    for (int i = 0; i < 1023; ++i) {
        for (int j = 0; j < 1023; j++) {
            // center the coordinates to get a proper fresnel term
            // divide by resolution which is in px/mm
            float xp = ((float)i / width - 0.5f) / resolution / (1.0) / d / lambda;	// mm 
            float yp = ((float)j / height - 0.5f) / resolution / (1.0) / d / lambda;	// mm

            //cout << xp << " " << yp << endl;
            float value = PI / (d * lambda) * (xp * xp + yp * yp); // (d * d * lambda * lambda);
            fresn.at<float>(i, j) = cos(value);
            fresn.at<float>(i + 1, j + 1) = sin(value);
        }
    }

    cv::normalize(fresn, fresn, 0, 1, cv::NORM_MINMAX);

    return fresn;
}

cv::Mat pupil() {
    cv::Mat pupi(1024, 1024, CV_32F);
    cv::Mat mask = cv::Mat::zeros(cv::Size(1024, 1024), CV_8UC1);
    cv::Mat gratings = cv::imread("C:\\Users\\acevedo\\Documents\\Visual Studio 2019\\Varjo\\ZedDepth - Copy\\gratings.png", 0);

    normalize(gratings, gratings, 0, 1, cv::NORM_MINMAX);
    gratings.convertTo(gratings, CV_32F);
    cv::Point ppl;

    pupi = pupi + gratings;
    ppl.x = 512;
    ppl.y = 512;

    circle(mask, ppl, 400, cv::Scalar(255), cv::FILLED, cv::LINE_AA);

    pupi = gratings + pupi;
    for (int i = 0; i < 750; ++i) {
        cv::Point org;
        org.x = rand() % 1024 + 1;
        org.y = rand() % 1024 + 1;
        circle(pupi, org, 5, cv::Scalar(0), cv::FILLED, cv::LINE_AA);
    }

    mask.convertTo(mask, CV_32F);
    pupi = pupi.mul(mask);
    cv::Mat res;
    bitwise_not(pupi, res);


    imshow("gratings", res);
    return pupi;
}

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

void createKernel(int kernelRadl) {
    std::vector<float> gauss_vec;
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
}

cv::Mat RGB4Lambda(int samples, cv::Mat magnitude,cv::Mat dst2, bool colored) {
    for (int l = 0; l < samples; ++l) {
        float lambdai = l * 390 / samples;
        lambdai = 380 + lambdai;

        float lambdai_idx = lambdai - 380;
        int idx = (int)lambdai_idx;

        float X1 = spectrum[idx * 3];
        float Y1 = spectrum[idx * 3 + 1];
        float Z1 = spectrum[idx * 3 + 2];

        idx = idx + 1;
        float X2 = spectrum[idx * 3];
        float Y2 = spectrum[idx * 3 + 1];
        float Z2 = spectrum[idx * 3 + 2];

        float X = (X2 - X1) * (lambdai_idx - idx) + X1;
        float Y = (Y2 - Y1) * (lambdai_idx - idx) + Y1;
        float Z = (Z2 - Z1) * (lambdai_idx - idx) + Z1;

        float scale = 575 / lambdai;

        cv::Vec3f xyz = { X, Y, Z };
        cv::Mat_<float> dst(floor(magnitude.rows / scale), floor(magnitude.rows / scale));

        cv::Vec3f tmp;
        xyz = xyz / 100;
        cv::Mat_<float> intensity(1348, 1348);
        float mat[3][3] = {
        3.2406, -1.5372, -0.4986,
        -0.9689, 1.8758, 0.0415,
        0.0557, -0.2040, 1.0570
        };

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; j++) {
                tmp[i] += mat[i][j] * xyz[j];
            }
        }

        cv::Vec3f bgr;
        bgr[2] = (tmp[0] > 0.0031308) ? ((1.055 * pow(tmp[0], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[0]);
        bgr[1] = (tmp[1] > 0.0031308) ? ((1.055 * pow(tmp[1], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[1]);
        bgr[0] = (tmp[2] > 0.0031308) ? ((1.055 * pow(tmp[2], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[2]);

        bgr *= 255.0;


        for (int x = 0;x < dst.rows; x++) {

            for (int y = 0; y < dst.cols; y++) {

                float xi = (float)x * (float)575 / lambdai;
                float yi = (float)y * (float)575 / lambdai;


                float x_pos, y_pos, intpX, intpY;

                intpX = modf((float)xi, &x_pos);
                intpY = modf((float)yi, &y_pos);

                float pix1,pix2, pix3, pix4;

                if (x_pos > 0 && x_pos < magnitude.rows - 1 && y_pos > 0 && y_pos <magnitude.cols - 1 && x < 1024 && y < 1024) {
                    pix1 = magnitude.at<float>(x_pos, y_pos);
                    pix2 = magnitude.at<float>((x_pos + 1), y_pos);
                    pix3 = magnitude.at<float>(x_pos, (y_pos + 1));
                    pix4 = magnitude.at<float>((x_pos + 1), (y_pos + 1));
                    dst.at<float>(x, y) = interpolate_pix(pix1, pix2, pix3, pix4, intpX, intpY, bgr);
                }
            }
        }

        int w = (intensity.cols - dst.cols) / 2;
        int z = (intensity.rows - dst.rows) / 2;
        
        #pragma omp parallel for 
        for (int intx = 0; intx < dst.rows - 1; intx++) {
            for (int inty = 0; inty < dst.cols - 1; inty++) {
                    intensity.at<float>(intx + w, inty + z) += dst.at<float>(intx, inty);
                      dst2.at<cv::Vec3f>(intx + w, inty + z)[0] += dst.at<float>(intx, inty) * bgr[0] ;
                      dst2.at<cv::Vec3f>(intx + w, inty + z)[1] += dst.at<float>(intx, inty) * bgr[1];
                      dst2.at<cv::Vec3f>(intx + w, inty + z)[2] += dst.at<float>(intx, inty) * bgr[2];
            }
        
            
        }
        
    }
    return dst2;
}

cv::Mat spectMultiplication(cv::Mat h_DFT_img, cv::Mat h_dftcc[], cv::Mat res[], cv::Mat imgout[], int new_width, int new_height, cv::Mat reschannel[]) {
    #pragma omp parallel for
    for(int i=0;i<3;++i){

    cv::mulSpectrums(h_DFT_img, h_dftcc[i], res[i], 0);
    cv::dft(res[i], imgout[i], cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    fftshift(imgout[i], imgout[i]);
    normalize(imgout[i], imgout[i], 0, 255, cv::NORM_MINMAX);
    imgout[i].convertTo(imgout[i], CV_8UC1);
    cv::resize(imgout[i], imgout[i], cv::Size(new_width, new_height), cv::INTER_LINEAR);
    reschannel[i] = reschannel[i] + imgout[i];

    }
    return reschannel[3];
}


/*
cv::Mat temporalGlare(cv::Mat zedImg, cv::cuda::GpuMat d_threshold, int lightSensitivity, bool cudaglare, cv::Mat h_dftcc[],cv::Mat_<float> h_dftoc,bool colored) {

        cv::Mat reschannel[4];
        cv::split(zedImg, reschannel);

        cv::cuda::GpuMat tmp1, tmp2, tmp3;
        cv::cuda::cvtColor(d_threshold, tmp1, cv::COLOR_BGRA2GRAY);
        cv::cuda::threshold(tmp1, tmp2, lightSensitivity, 255, cv::THRESH_TOZERO);

        cv::Mat_<float> tp3;
        cv::cuda::normalize(tmp2, tmp3, 0, 1, cv::NORM_MINMAX, -1);
        tmp3.convertTo(tmp3, CV_32F);

        cv::Mat imgout[3], res[3];
        cv::Mat h_DFT_img;
        //if (cudaglare) {
        //
        //    cv::cuda::GpuMat d_DFT_img;
        //    cv::Mat h_DFT_img, h_DFT_psf;
        //    cv::cuda::GpuMat dres, dres1, dres2, dres3, dimgout, dimgout1, dimgout2, dimgout3;
        //
        //
        //    cv::cuda::resize(tmp3, tmp3, cv::Size(h_dftcc[0].rows, h_dftcc[0].cols), cv::INTER_LINEAR);
        //
        //    cv::cuda::dft(tmp3, d_DFT_img, tmp3.size(), 0);
        //
        //    cv::cuda::multiply(d_DFT_img, h_dftcc[0], dres1, 1, -1);
        //    cv::cuda::multiply(d_DFT_img, h_dftcc[1], dres2, 1, -1);
        //    cv::cuda::multiply(d_DFT_img, h_dftcc[2], dres3, 1, -1);
        //
        //    cv::Size dest_size = tmp3.size();
        //    int flag = cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE;
        //    cv::cuda::dft(dres1, dimgout1, dest_size, flag);
        //    cv::cuda::dft(dres2, dimgout2, dest_size, flag);
        //    cv::cuda::dft(dres3, dimgout3, dest_size, flag);
        //
        //
        //    dimgout1.download(imgout[0]);
        //    dimgout2.download(imgout[1]);
        //    dimgout3.download(imgout[2]);
        //
        //}
        //else {
            cv::Mat h_threshold;
            tmp3.download(h_threshold);
        
            h_threshold.convertTo(h_threshold, CV_32F);
            cv::Mat_<float> tmp4;
            resize(h_threshold, tmp4, h_dftcc[0].size(), cv::INTER_NEAREST);
            calculateDFT(tmp4, h_DFT_img);
           
        
        

            if (colored) {
                std::cout << "pinga" << std::endl;

                spectMultiplication(h_DFT_img, h_dftcc, res, imgout, zedImg.rows, zedImg.cols, reschannel);
            }
            else {
                

                cv::mulSpectrums(h_DFT_img, h_dftoc, res[0], 0);
                dft(res[0], imgout[0], cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
                fftshift(imgout[0], imgout[0]);
                normalize(imgout[0], imgout[0], 0, 255, cv::NORM_MINMAX);
                imgout[0].convertTo(imgout[0], CV_8UC1);
                cv::resize(imgout[0], imgout[0], zedImg.size(), cv::INTER_LINEAR);
                reschannel[0] = reschannel[0] + imgout[0];
                reschannel[1] = reschannel[1] + imgout[0];
                   
                reschannel[2] = reschannel[2] + imgout[0];
            }
        
        
        
            reschannel[3] = 255;
            std::vector<cv::Mat> RESchannels{ reschannel[0], reschannel[1], reschannel[2], reschannel[3] };
            cv::merge(RESchannels, zedImg);
            cv::imshow("Glare draft", zedImg);
        //}
        return zedImg;
}
*/
cv::cuda::GpuMat distortionMapsgpu(cv::cuda::GpuMat view, int fy, int fx, int radius, cv::Mat srcX, cv::Mat srcY, cv::Mat srcX2, cv::Mat srcY2, float distIntens) {
    float distortionx=0;
    float distortiony=0;
    cv::cuda::GpuMat res1;
    cv::cuda::GpuMat res2;
    

    #pragma omp parallel for 
    for (int i = 0; i < view.rows; ++i) {
        for (int j = 0; j < view.cols; j++) {
            auto dx = i - fy;
            auto dy = j - fx;


            auto dis = dx * dx + dy * dy;
            if (dis >= radius * radius) {
                srcX.at<float>(i, j) = (j);
                srcY.at<float>(i, j) = (i);
                srcX2.at<float>(i, j) = j;
                srcY2.at<float>(i, j) = i;
            }
            else {
                double factor = 1.0;
                if (dis > 0.0) {
                    factor = pow(sin(3.141592 * sqrtf(dis) / radius / 2.), distIntens);
                    distortionx = ((distIntens * 10) * sqrtf(dis) / radius) * sin(i / 15.0);
                    distortiony = ((distIntens * 10) * sqrtf(dis) / radius) * cos(j / 15.0);
                }
                srcX.at<float>(i, j) = static_cast<float>(factor * dy / 1.0 + fx);
                srcY.at<float>(i, j) = static_cast<float>(factor * dx / 1.0 + fy);
                srcX2.at<float>(i, j) = j + distortionx;
                srcY2.at<float>(i, j) = i + distortiony;
            }

        }
    }
    cv::cuda::GpuMat d_srcx, d_srcy;
    d_srcx.upload(srcX);
    d_srcy.upload(srcY);
    
    cv::cuda::GpuMat d_srcx2, d_srcy2;
    d_srcx2.upload(srcX2);
    d_srcy2.upload(srcY2);
    //
    cv::cuda::remap(view, res1, d_srcx, d_srcy, cv::INTER_NEAREST, cv::BORDER_REPLICATE);
    cv::cuda::remap(res1, res2, d_srcx2, d_srcy2, cv::INTER_NEAREST, cv::BORDER_REPLICATE);
    return res2;
}

cv::Mat distortionMaps(cv::Mat view, int fy, int fx, int radius, float distIntens) {
    float distortionx = 0;
    float distortiony = 0;
    cv::Mat res1;
    cv::Mat res2;

    cv::Mat srcY2(view.rows, view.cols, CV_32F);
    cv::Mat srcX2(view.rows, view.cols, CV_32F);
    cv::Mat srcX(view.rows,  view.cols, CV_32F);
    cv::Mat srcY(view.rows,  view.cols, CV_32F);


#pragma omp parallel for 
    for (int i = 0; i < view.rows; ++i) {
        for (int j = 0; j < view.cols; j++) {
            auto dx = i - fy;
            auto dy = j - fx;


            auto dis = dx * dx + dy * dy;
            if (dis >= radius * radius) {
                srcX.at<float>(i, j) = (j);
                srcY.at<float>(i, j) = (i);
                srcX2.at<float>(i, j) = j;
                srcY2.at<float>(i, j) = i;
            }
            else {
                double factor = 1.0;
                if (dis > 0.0) {
                    factor = pow(sin(3.141592 * sqrtf(dis) / radius / 2.), distIntens);
                    distortionx = ((distIntens * 10) * sqrtf(dis) / radius) * sin(i / 15.0);
                    distortiony = ((distIntens * 10) * sqrtf(dis) / radius) * cos(j / 15.0);
                }
                srcX.at<float>(i, j) = static_cast<float>(factor * dy / 1.0 + fx);
                srcY.at<float>(i, j) = static_cast<float>(factor * dx / 1.0 + fy);
                srcX2.at<float>(i, j) = j + distortionx;
                srcY2.at<float>(i, j) = i + distortiony;
            }

        }
    }
    cv::remap(view, res1, srcX, srcY, cv::INTER_NEAREST, cv::BORDER_REPLICATE);
    cv::remap(res1, res2, srcX2, srcY2, cv::INTER_NEAREST, cv::BORDER_REPLICATE);
    //cv::imshow("temp", res2);
    return res2;
}