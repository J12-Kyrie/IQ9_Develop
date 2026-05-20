#ifndef QWEN_VL_OPENCL_ACCELERATOR_H
#define QWEN_VL_OPENCL_ACCELERATOR_H

#include <CL/cl.h>
#include <vector>

namespace qwen_vl {

// Forward declaration
struct QwenVLConfig;

class OpenCLAccelerator {
public:
    OpenCLAccelerator();
    ~OpenCLAccelerator();

    bool initialize();
    void setConfig(const QwenVLConfig* config);
    void printDeviceInfo();
    
    void normalizeImage(const uint8_t* src, float* dst,
                       int32_t height, int32_t width, int32_t channels,
                       const float* mean, const float* std);
    
    void transposeToPatch(const float* normalized_image, float* output_patches,
                         int32_t T, int32_t H, int32_t W, int32_t C,
                         int32_t patch_size, int32_t merge_size,
                         int32_t temporal_patch_size, int32_t input_dim);
    
    // UPDATED: Add missing parameters to match implementation
    void computeRotaryPosEmb(
        const int32_t* grid_thw, float* pos_cos, float* pos_sin,
        int32_t seq_len, int32_t temporal_dim, int32_t height_dim, 
        int32_t width_dim, float rope_theta);
    
    void initAttentionMasks(float* full_mask, float* window_mask, int32_t seq_len);
    
private:
    bool createProgram();
    const char* getErrorString(cl_int error);

    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    cl_kernel normalize_kernel_;
    cl_kernel transpose_kernel_;
    cl_kernel rotary_kernel_;
    cl_kernel mask_kernel_;
    bool initialized_;
    const QwenVLConfig* config_;
};

} // namespace qwen_vl

#endif