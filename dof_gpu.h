#ifndef DOF_GPU_H
#define DOF_GPU_H

/* dof_gpu.h.
 *
 * This file contains the interface to the CUDA functions ,
 * for rendering depth of field, based on Gaussian blurring
 * using separable convolution, with depth-dependent kernel size.
 * Separable convolution is based on convolution CUDA Sample with kernel-size adaptation
 */

 // ZED includes
#include <sl/Camera.hpp>

//#define KERNEL_RADIUS 32 //see assert in convolution kernel to see the limitations of size
#define KERNEL_LENGTH_X(x) (2 * x + 1)
#define MAX_KERNEL_LENGTH KERNEL_LENGTH(MAX_KERNEL_RADIUS)
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

 // Copy gaussien kernel into GPU memory
void copyKernel(float* kernel_coefficients, int kernel_index);
void testKernel();

// Normalize depth between 0.f and 1.f
void normalizeDepth(float* depth, float* depth_out, unsigned int step, float min_distance, float max_distance, unsigned int width, unsigned height);

// GPU convolution
void convolutionRows(sl::uchar4* d_Dst, sl::uchar4* d_Src, float* i_depth, int imageW, int imageH, int depth_pitch, float focus_point, int kernelRad, bool full,float*g, int gpitch);
void convolutionColumns(sl::uchar4* d_Dst, sl::uchar4* d_Src, float* i_depth, int imageW, int imageH, int depth_pitch, float focus_point, int kernelRad, bool full,float* g,int gpitch);

void contrast(sl::uchar4* src, sl::uchar4* dst, sl::uchar4* dst2, int imageW, int imageH, unsigned int step,  float p, float* g, int pitchdepth);
void vignetteRows(sl::uchar4* d_Dst, sl::uchar4* d_Src, float* i_depth, int imageW, int imageH, int depth_pitch);

void convolutionpsfColumns(cv::cuda::GpuMat& d_Dst, cv::cuda::GpuMat& d_Src, cv::cuda::GpuMat& d_Src2, int imageW, int imageH, int kernelRad);


#endif //DOF_GPU_H