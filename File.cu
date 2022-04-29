/*
 * This file contains the definition of the CUDA functions ,
 * for rendering depth of field, based on Gaussian blurring
 * using separable convolution, with depth-dependent kernel size.
 * Separable convolution is based on convolution CUDA Sample with kernel-size adaptation
*/

#include "dof_gpu.h"

__constant__ float c_kernel[32 * (32 + 2)];

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void copyKernel(float* kernel_coefficients, int kernel_index) {
    int kernel_radius = kernel_index + 1;
    cudaMemcpyToSymbol(c_kernel, kernel_coefficients,
        KERNEL_LENGTH_X(kernel_radius) * sizeof(float),
        kernel_index * (kernel_index + 2) * sizeof(float));
}

void testKernel() {
    float h_kernel_data[32 * (32 + 2)];
    cudaMemcpyFromSymbol(h_kernel_data, c_kernel, 32 * (32 + 2) * sizeof(float));
    int i, j;
    for (i = 0; i < 32; ++i) {
        printf("%d: ", i);
        for (j = 0; j < 2 * i + 3; ++j)
            printf("%f ", h_kernel_data[i * (i + 2) + j]);
        printf("\n");
    }
}


__global__ void _k_normalizeDepth(float* depth, float* depth_norm, unsigned int step, float min_distance, float max_distance, unsigned int width, unsigned height) {
    uint32_t x_local = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y_local = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_local >= width || y_local >= height) return;

    float depth_world = depth[x_local + y_local * step];
    float depth_normalized = (max_distance - depth_world) / (max_distance - min_distance);

    if (depth_normalized < 0.f) depth_normalized = 0.f;
    if (depth_normalized > 1.f) depth_normalized = 1.f;

    if (isfinite(depth_normalized))
        depth_norm[x_local + y_local * step] = depth_normalized;
}

void normalizeDepth(float* depth, float* depth_out, unsigned int step, float min_distance, float max_distance, unsigned int width, unsigned height) {
    dim3 dimGrid, dimBlock;

    dimBlock.x = 32;
    dimBlock.y = 8;

    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

    _k_normalizeDepth << <dimGrid, dimBlock, 0 >> > (depth, depth_out, step, min_distance, max_distance, width, height);
}



#define   ROWS_BLOCKDIM_X 32
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

__global__ void _k_convolutionRows(sl::uchar4* d_Dst, sl::uchar4* d_Src, float* depth, int imageW, int imageH, int pitch, int pitch_depth, float focus_depth, int KERNEL_RADIUS) {
    __shared__ sl::uchar4 s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;
    depth += baseY * pitch_depth + baseX;

    sl::uchar4 reset(0, 0, 0, 0);
    //Load main data
#pragma unroll
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    //Load left halo
#pragma unroll
    for (int i = 0; i < ROWS_HALO_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : reset;
    }

    //Load right halo
#pragma unroll
    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : reset;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        sl::float3 sum(0, 0, 0);
        int kernel_radius = floorf((KERNEL_RADIUS)*fabs(depth[i * ROWS_BLOCKDIM_X] - focus_depth));
        int kernel_mid = kernel_radius * kernel_radius - 1 + kernel_radius;
        if (kernel_radius > 0) {
            for (int j = -kernel_radius; j <= kernel_radius; ++j) {
                sum.x += c_kernel[kernel_mid + j] * (float)s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j].x;
                sum.y += c_kernel[kernel_mid + j] * (float)s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j].y;
                sum.z += c_kernel[kernel_mid + j] * (float)s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j].z;
            }
        }
        else {
            sum.x = (float)s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X].x;
            sum.y = (float)s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X].y;
            sum.z = (float)s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X].z;
        }

        d_Dst[i * ROWS_BLOCKDIM_X].x = sum.x;
        d_Dst[i * ROWS_BLOCKDIM_X].y = sum.y;
        d_Dst[i * ROWS_BLOCKDIM_X].z = sum.z;
        d_Dst[i * ROWS_BLOCKDIM_X].w = 255;
    }
}

void convolutionRows(sl::uchar4* d_Dst, sl::uchar4* d_Src, float* i_depth, int imageW, int imageH, int depth_pitch, float focus_point,int kernelRad) {
    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
    _k_convolutionRows << <blocks, threads >> > (d_Dst, d_Src, i_depth, imageW, imageH, imageW, depth_pitch, focus_point, kernelRad);
}


#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 2
#define   COLUMNS_HALO_STEPS 4

__global__ void _k_convolutionColumns(sl::uchar4* d_Dst, sl::uchar4* d_Src, float* depth, int imageW, int imageH, int pitch, int pitch_depth, float focus_depth, int KERNEL_RADIUS) {
    __shared__ sl::uchar4 s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];
    sl::uchar4 reset(0, 0, 0, 0);
    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;
    depth += baseY * pitch_depth + baseX;

    //Main data
#pragma unroll
    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    //Upper halo
#pragma unroll
    for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : reset;
    }

    //Lower halo
#pragma unroll
    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : reset;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll
    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        sl::float3 sum(0, 0, 0);
        int kernel_radius = floorf((KERNEL_RADIUS)*fabs(depth[i * COLUMNS_BLOCKDIM_Y * pitch] - focus_depth));
        int kernel_mid = kernel_radius * kernel_radius - 1 + kernel_radius;

        if (kernel_radius > 0) {
            for (int j = -kernel_radius; j <= kernel_radius; ++j) {
                sum.x += c_kernel[kernel_mid + j] * (float)s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j].z;
                sum.y += c_kernel[kernel_mid + j] * (float)s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j].y;
                sum.z += c_kernel[kernel_mid + j] * (float)s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j].x;
            }
        }
        else {
            sum.x = (float)s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y].z;
            sum.y = (float)s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y].y;
            sum.z = (float)s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y].x;
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].x = sum.x;
        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].y = sum.y;
        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].z = sum.z;
        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].w = 255;
    }
}

void convolutionColumns(sl::uchar4* d_Dst, sl::uchar4* d_Src, float* i_depth, int imageW, int imageH, int depth_pitch, float focus_point, int kernelRad) {
    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    _k_convolutionColumns << <blocks, threads >> > (d_Dst, d_Src, i_depth, imageW, imageH, imageW, depth_pitch, focus_point,  kernelRad);
}

//#define angle 0

__device__ float3 bgr2xyz(uchar3 src) {
    float scr_b = src.x / 255.0;
    float scr_g = src.y / 255.0;
    float scr_r = src.z / 255.0;

    float tmp[3];
    tmp[0] = 100.0 * ((scr_r > .04045) ? pow((scr_r + .055) / 1.055, 2.4) : scr_r / 12.92);
    tmp[1] = 100.0 * ((scr_g > .04045) ? pow((scr_g + .055) / 1.055, 2.4) : scr_g / 12.92);
    tmp[2] = 100.0 * ((scr_b > .04045) ? pow((scr_b + .055) / 1.055, 2.4) : scr_b / 12.92);

    float3 xyz;
    xyz.x = .4124 * tmp[0] + .3576 * tmp[1] + .1805 * tmp[2];
    xyz.y = .2126 * tmp[0] + .7152 * tmp[1] + .0722 * tmp[2];
    xyz.z = .0193 * tmp[0] + .1192 * tmp[1] + .9505 * tmp[2];

    return xyz;
}

__device__ float3 xyz2lab(float3 src, float p) {

    float scr_z = src.z / 108.883;
    float scr_y = src.y / 100.;
    float scr_x = src.x / 95.047;

    float PI = 3.14159265358979323846;

    float3 v;
    v.x = (scr_x > .008856) ? pow(scr_x, 1. / 3.) : (7.787 * scr_x) + (16. / 116.);
    v.y = (scr_y > .008856) ? pow(scr_y, 1. / 3.) : (7.787 * scr_y) + (16. / 116.);
    v.z = (scr_z > .008856) ? pow(scr_z, 1. / 3.) : (7.787 * scr_z) + (16. / 116.);

    float3 lab;
    lab.x = (116. * v.y) - 16.;
    lab.y = 500. * (v.x - v.y);
    lab.z = 200. * (v.y - v.z);

    //float C = sqrt(pow(lab.y, 2) + pow(lab.z, 2));
    //float h = atan2(lab.z, lab.y);
    //h += (angle * PI) / 180.0;
    //lab.y = cos(h) * C;
    //lab.z = sin(h) * C;

    lab.x = lab.x * p + 50 * (1 - p);

    return lab;
}

__device__ float3 bgr2lab(uchar3 c, float p) {
    return xyz2lab(bgr2xyz(c), p);
}

__device__ float3 lab2xyz(float3 src) {

    float fy = (src.x + 16.0) / 116.0;
    float fx = src.y / 500.0 + fy;
    float fz = fy - src.z / 200.0;

    float3 lab;
    lab.x = 95.047 * ((fx > 0.206897) ? fx * fx * fx : (fx - 16.0 / 116.0) / 7.787);
    lab.y = 100.000 * ((fy > 0.206897) ? fy * fy * fy : (fy - 16.0 / 116.0) / 7.787);
    lab.z = 108.883 * ((fz > 0.206897) ? fz * fz * fz : (fz - 16.0 / 116.0) / 7.787);

    return lab;
}

__device__ float3 xyz2bgr(float3 src) {

    src.x /= 100.0;
    src.y /= 100.0;
    src.z /= 100.0;


    float tmp[3];

    tmp[0] = 3.2406 * src.x - 1.5372 * src.y - 0.4986 * src.z;
    tmp[1] = -0.9689 * src.x + 1.8758 * src.y + 0.0415 * src.z;
    tmp[2] = 0.0557 * src.x - 0.2040 * src.y + 1.0570 * src.z;

    float3 bgr;
    bgr.z = 255.0 * ((tmp[0] > 0.0031308) ? ((1.055 * pow(tmp[0], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[0]));
    bgr.y = 255.0 * ((tmp[1] > 0.0031308) ? ((1.055 * pow(tmp[1], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[1]));
    bgr.x = 255.0 * ((tmp[2] > 0.0031308) ? ((1.055 * pow(tmp[2], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[2]));

    return bgr;
}

__device__ float3 lab2bgr(float3 src) {
    return xyz2bgr(lab2xyz(src));
}



__global__ void k_contrast(const sl::uchar4* d_Src, sl::uchar4* d_Dst, int imageW, int imageH, int pitch, float p)
{

    __shared__ sl::uchar4 s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];
    sl::uchar4 reset(0, 0, 0, 0);
    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;


    //Main data
#pragma unroll
    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    //Upper halo
#pragma unroll
    for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : reset;
    }

    //Lower halo
#pragma unroll
    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : reset;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll
    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {


        uchar3 bgr;
        float3 transformed;

        bgr.x = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch].x;
        bgr.y = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch].y;
        bgr.z = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch].z;

   

        transformed.x = lab2bgr(bgr2lab(bgr, p)).x;
        transformed.y = lab2bgr(bgr2lab(bgr, p)).y;
        transformed.z = lab2bgr(bgr2lab(bgr, p)).z;
            
        

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].x = (unsigned char)(transformed.x < 0 ? 0 : (transformed.x > 255 ? 255 : transformed.x));
        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].y = (unsigned char)(transformed.y < 0 ? 0 : (transformed.y > 255 ? 255 : transformed.y));
        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].z = (unsigned char)(transformed.z < 0 ? 0 : (transformed.z > 255 ? 255 : transformed.z));
        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch].w = 255;
    }
}

   
        

      


void contrast(sl::uchar4 * src, sl::uchar4 * dst, int imageW, int imageH, unsigned int step, float p)
{
    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    k_contrast <<< blocks,threads >>> (src, dst, imageW, imageH, step, p);
    

}

