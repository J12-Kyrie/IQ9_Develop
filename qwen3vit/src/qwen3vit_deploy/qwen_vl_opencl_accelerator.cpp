/*
 * OpenCL Accelerator Implementation
 * Fixed: Removed CL_MEM_USE_HOST_PTR, added proper RoPE dimensions
 */

#include "qwen_vl_opencl_accelerator.h"
#include "qwen_vl_preprocessor.h"
#include "qwen_vl_opencl_kernels.h"
#include <iostream>
#include <cstring>
#include <chrono>

namespace qwen_vl {

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << "Error: " << msg << " - " << getErrorString(err) << std::endl; \
        return; \
    }

#define CHECK_ERROR_RET(err, msg, ret) \
    if (err != CL_SUCCESS) { \
        std::cerr << "Error: " << msg << " - " << getErrorString(err) << std::endl; \
        return ret; \
    }

OpenCLAccelerator::OpenCLAccelerator()
    : platform_(nullptr), device_(nullptr), context_(nullptr),
      queue_(nullptr), program_(nullptr),
      normalize_kernel_(nullptr), transpose_kernel_(nullptr),
      rotary_kernel_(nullptr), mask_kernel_(nullptr),
      initialized_(false), config_(nullptr) {
}

void OpenCLAccelerator::setConfig(const QwenVLConfig* config) {
    config_ = config;
}

OpenCLAccelerator::~OpenCLAccelerator() {
    if (normalize_kernel_) clReleaseKernel(normalize_kernel_);
    if (transpose_kernel_) clReleaseKernel(transpose_kernel_);
    if (rotary_kernel_) clReleaseKernel(rotary_kernel_);
    if (mask_kernel_) clReleaseKernel(mask_kernel_);
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}

const char* OpenCLAccelerator::getErrorString(cl_int error) {
    switch(error) {
        case CL_SUCCESS: return "Success";
        case CL_DEVICE_NOT_FOUND: return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
        case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case CL_OUT_OF_RESOURCES: return "Out of resources";
        case CL_BUILD_PROGRAM_FAILURE: return "Build program failure";
        case CL_INVALID_PROGRAM: return "Invalid program";
        case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
        case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
        case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
        case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
        case CL_INVALID_HOST_PTR: return "Invalid host pointer";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory allocation failure";
        default: return "Unknown error";
    }
}

void OpenCLAccelerator::printDeviceInfo() {
    char device_name[128];
    char device_vendor[128];
    char device_version[128];
    cl_ulong global_mem_size;
    cl_ulong local_mem_size;
    size_t max_work_group_size;
    
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(device_version), device_version, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
    
    std::cout << "=== OpenCL Device Information ===" << std::endl;
    std::cout << "Device: " << device_name << std::endl;
    std::cout << "Vendor: " << device_vendor << std::endl;
    std::cout << "Version: " << device_version << std::endl;
    std::cout << "Global Memory: " << (global_mem_size / (1024*1024)) << " MB" << std::endl;
    std::cout << "Local Memory: " << (local_mem_size / 1024) << " KB" << std::endl;
    std::cout << "Max Work Group Size: " << max_work_group_size << std::endl;
    
    // Check for Qualcomm extensions
    size_t ext_size;
    clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
    std::vector<char> extensions(ext_size);
    clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, ext_size, extensions.data(), nullptr);
    
    std::string ext_str(extensions.data());
    if (ext_str.find("cl_qcom") != std::string::npos) {
        std::cout << "✓ Qualcomm extensions detected" << std::endl;
    }
    if (ext_str.find("cl_khr_fp16") != std::string::npos) {
        std::cout << "✓ FP16 support available" << std::endl;
    }
    std::cout << "=================================" << std::endl;
}

bool OpenCLAccelerator::initialize() {
    cl_int err;
    
    // Get platform
    err = clGetPlatformIDs(1, &platform_, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL platform: " << getErrorString(err) << std::endl;
        return false;
    }
    
    // Get GPU device (prefer GPU for Adreno)
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "GPU not found, trying default device..." << std::endl;
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_DEFAULT, 1, &device_, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to get OpenCL device: " << getErrorString(err) << std::endl;
            return false;
        }
    }
    
    // Print device info
    printDeviceInfo();
    
    // Create context
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create context: " << getErrorString(err) << std::endl;
        return false;
    }
    
    // Create command queue
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue: " << getErrorString(err) << std::endl;
        return false;
    }
    
    // Create and build program
    if (!createProgram()) {
        return false;
    }
    
    initialized_ = true;
    std::cout << "✅ OpenCL initialized successfully!" << std::endl;
    return true;
}

bool OpenCLAccelerator::createProgram() {
    cl_int err;
    
    // Create program from source
    const char* source = OPENCL_KERNEL_SOURCE;
    size_t source_size = strlen(source);
    
    program_ = clCreateProgramWithSource(context_, 1, &source, &source_size, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program: " << getErrorString(err) << std::endl;
        return false;
    }
    
    // Build program with optimizations for Adreno
    const char* build_options = "-cl-fast-relaxed-math -cl-mad-enable";
    err = clBuildProgram(program_, 1, &device_, build_options, nullptr, nullptr);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to build program: " << getErrorString(err) << std::endl;
        
        // Print build log
        size_t log_size;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        return false;
    }
    
    // Create kernels
    normalize_kernel_ = clCreateKernel(program_, "normalize_image", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create normalize kernel: " << getErrorString(err) << std::endl;
        return false;
    }
    
    transpose_kernel_ = clCreateKernel(program_, "transpose_to_patch", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create transpose kernel: " << getErrorString(err) << std::endl;
        return false;
    }
    
    rotary_kernel_ = clCreateKernel(program_, "compute_rotary_emb", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create rotary kernel: " << getErrorString(err) << std::endl;
        return false;
    }
    
    mask_kernel_ = clCreateKernel(program_, "init_attention_masks", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create mask kernel: " << getErrorString(err) << std::endl;
        return false;
    }
    
    std::cout << "✅ All kernels created successfully" << std::endl;
    return true;
}

void OpenCLAccelerator::normalizeImage(
    const uint8_t* src, float* dst,
    int32_t height, int32_t width, int32_t channels,
    const float* mean, const float* std) {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cl_int err;
    size_t data_size = height * width * channels;
    
    // Create buffers on device
    cl_mem src_buf = clCreateBuffer(context_, 
        CL_MEM_READ_ONLY,
        data_size * sizeof(uint8_t), nullptr, &err);
    CHECK_ERROR(err, "Failed to create src buffer");
    
    cl_mem dst_buf = clCreateBuffer(context_,
        CL_MEM_WRITE_ONLY,
        data_size * sizeof(float), nullptr, &err);
    CHECK_ERROR(err, "Failed to create dst buffer");
    
    // Write input data to device
    err = clEnqueueWriteBuffer(queue_, src_buf, CL_TRUE, 0,
                               data_size * sizeof(uint8_t), src, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to write src buffer");
    
    // Set kernel arguments
    clSetKernelArg(normalize_kernel_, 0, sizeof(cl_mem), &src_buf);
    clSetKernelArg(normalize_kernel_, 1, sizeof(cl_mem), &dst_buf);
    clSetKernelArg(normalize_kernel_, 2, sizeof(float), &mean[0]);
    clSetKernelArg(normalize_kernel_, 3, sizeof(float), &mean[1]);
    clSetKernelArg(normalize_kernel_, 4, sizeof(float), &mean[2]);
    clSetKernelArg(normalize_kernel_, 5, sizeof(float), &std[0]);
    clSetKernelArg(normalize_kernel_, 6, sizeof(float), &std[1]);
    clSetKernelArg(normalize_kernel_, 7, sizeof(float), &std[2]);
    clSetKernelArg(normalize_kernel_, 8, sizeof(int32_t), &height);
    clSetKernelArg(normalize_kernel_, 9, sizeof(int32_t), &width);
    clSetKernelArg(normalize_kernel_, 10, sizeof(int32_t), &channels);
    
    // Execute kernel
    size_t global_size[2] = {(size_t)height, (size_t)width};
    err = clEnqueueNDRangeKernel(queue_, normalize_kernel_, 2, nullptr,
                                 global_size, nullptr, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to execute normalize kernel");
    
    // Read output data back to host
    err = clEnqueueReadBuffer(queue_, dst_buf, CL_TRUE, 0,
                              data_size * sizeof(float), dst, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to read dst buffer");
    
    // Cleanup
    clReleaseMemObject(src_buf);
    clReleaseMemObject(dst_buf);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  OpenCL normalize: " << duration.count() / 1000.0 << " ms" << std::endl;
}

void OpenCLAccelerator::transposeToPatch(
    const float* normalized_image, float* output_patches,
    int32_t T, int32_t H, int32_t W, int32_t C,
    int32_t patch_size, int32_t merge_size, 
    int32_t temporal_patch_size, int32_t input_dim) {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cl_int err;
    
    // Calculate sizes
    int32_t gridT = T / temporal_patch_size;
    int32_t gridH = H / patch_size;
    int32_t gridW = W / patch_size;
    int32_t merged_gridH = gridH / merge_size;
    int32_t merged_gridW = gridW / merge_size;
    int32_t seq_length = gridT * merged_gridH * merged_gridW * merge_size * merge_size;
    
    size_t input_size = T * H * W * C * sizeof(float);
    size_t output_size = seq_length * input_dim * sizeof(float);
    
    // Create buffers on device
    cl_mem input_buf = clCreateBuffer(context_,
        CL_MEM_READ_ONLY,
        input_size, nullptr, &err);
    CHECK_ERROR(err, "Failed to create input buffer");
    
    cl_mem output_buf = clCreateBuffer(context_,
        CL_MEM_WRITE_ONLY,
        output_size, nullptr, &err);
    CHECK_ERROR(err, "Failed to create output buffer");
    
    // Write input data to device
    err = clEnqueueWriteBuffer(queue_, input_buf, CL_TRUE, 0,
                               input_size, normalized_image, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to write input buffer");
    
    // Set kernel arguments
    clSetKernelArg(transpose_kernel_, 0, sizeof(cl_mem), &input_buf);
    clSetKernelArg(transpose_kernel_, 1, sizeof(cl_mem), &output_buf);
    clSetKernelArg(transpose_kernel_, 2, sizeof(int32_t), &T);
    clSetKernelArg(transpose_kernel_, 3, sizeof(int32_t), &H);
    clSetKernelArg(transpose_kernel_, 4, sizeof(int32_t), &W);
    clSetKernelArg(transpose_kernel_, 5, sizeof(int32_t), &C);
    clSetKernelArg(transpose_kernel_, 6, sizeof(int32_t), &patch_size);
    clSetKernelArg(transpose_kernel_, 7, sizeof(int32_t), &merge_size);
    clSetKernelArg(transpose_kernel_, 8, sizeof(int32_t), &temporal_patch_size);
    clSetKernelArg(transpose_kernel_, 9, sizeof(int32_t), &input_dim);
    
    // Execute kernel
    size_t global_size = seq_length;
    size_t local_size = 64;  // Tune this for your GPU
    err = clEnqueueNDRangeKernel(queue_, transpose_kernel_, 1, nullptr,
                                 &global_size, &local_size, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to execute transpose kernel");
    
    // Read output data back to host
    err = clEnqueueReadBuffer(queue_, output_buf, CL_TRUE, 0,
                              output_size, output_patches, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to read output buffer");
    
    // Cleanup
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  OpenCL transpose: " << duration.count() / 1000.0 << " ms" << std::endl;
}

void OpenCLAccelerator::computeRotaryPosEmb(
    const int32_t* grid_thw, float* pos_cos, float* pos_sin,
    int32_t seq_len, int32_t temporal_dim, int32_t height_dim, int32_t width_dim, float rope_theta)
{
    cl_int err;
    
    // Create buffers
    cl_mem grid_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     3 * sizeof(int32_t), (void*)grid_thw, &err);
    CHECK_ERROR(err, "Failed to create grid buffer");
    
    int32_t rope_dim = temporal_dim + height_dim + width_dim;
    cl_mem cos_buf = clCreateBuffer(context_, CL_MEM_WRITE_ONLY,
                                    seq_len * rope_dim * sizeof(float), nullptr, &err);
    CHECK_ERROR(err, "Failed to create cos buffer");
    
    cl_mem sin_buf = clCreateBuffer(context_, CL_MEM_WRITE_ONLY,
                                    seq_len * rope_dim * sizeof(float), nullptr, &err);
    CHECK_ERROR(err, "Failed to create sin buffer");
    
    // Set kernel arguments
    clSetKernelArg(rotary_kernel_, 0, sizeof(cl_mem), &grid_buf);
    clSetKernelArg(rotary_kernel_, 1, sizeof(cl_mem), &cos_buf);
    clSetKernelArg(rotary_kernel_, 2, sizeof(cl_mem), &sin_buf);
    clSetKernelArg(rotary_kernel_, 3, sizeof(int32_t), &seq_len);
    clSetKernelArg(rotary_kernel_, 4, sizeof(int32_t), &temporal_dim);
    clSetKernelArg(rotary_kernel_, 5, sizeof(int32_t), &height_dim);
    clSetKernelArg(rotary_kernel_, 6, sizeof(int32_t), &width_dim);
    clSetKernelArg(rotary_kernel_, 7, sizeof(float), &rope_theta);
    clSetKernelArg(rotary_kernel_, 8, sizeof(int32_t), &config_->merge_size);
    clSetKernelArg(rotary_kernel_, 9, sizeof(int32_t), &config_->temporal_patch_size);
    
    // Execute kernel
    size_t global_size[2] = {(size_t)seq_len, (size_t)rope_dim};
    err = clEnqueueNDRangeKernel(queue_, rotary_kernel_, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to execute rotary kernel");
    
    // Read results back
    err = clEnqueueReadBuffer(queue_, cos_buf, CL_TRUE, 0, seq_len * rope_dim * sizeof(float), pos_cos, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to read cos buffer");
    
    err = clEnqueueReadBuffer(queue_, sin_buf, CL_TRUE, 0, seq_len * rope_dim * sizeof(float), pos_sin, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to read sin buffer");
    
    // Cleanup
    clReleaseMemObject(grid_buf);
    clReleaseMemObject(cos_buf);
    clReleaseMemObject(sin_buf);
}

void OpenCLAccelerator::initAttentionMasks(
    float* full_mask, float* window_mask, int32_t seq_len) {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cl_int err;
    size_t mask_size = seq_len * seq_len * sizeof(float);
    
    // Initialize full_mask to all 0.0 on CPU (simple memset)
    std::memset(full_mask, 0, mask_size);
    
    // Create buffer for window mask on device
    cl_mem window_mask_buf = clCreateBuffer(context_,
        CL_MEM_WRITE_ONLY,
        mask_size, nullptr, &err);
    CHECK_ERROR(err, "Failed to create window mask buffer");
    
    // Initialize window mask to -1000.0 using kernel
    clSetKernelArg(mask_kernel_, 0, sizeof(cl_mem), &window_mask_buf);
    clSetKernelArg(mask_kernel_, 1, sizeof(int32_t), &seq_len);
    
    size_t global_size[2] = {(size_t)seq_len, (size_t)seq_len};
    err = clEnqueueNDRangeKernel(queue_, mask_kernel_, 2, nullptr,
                                 global_size, nullptr, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to execute mask kernel");
    
    // Read window mask back to host
    err = clEnqueueReadBuffer(queue_, window_mask_buf, CL_TRUE, 0,
                              mask_size, window_mask, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to read window mask buffer");
    
    // Apply windowed regions (set to 0.0) on CPU
    std::vector<int32_t> cu_window_seqlens = {
        0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024
    };
    for (size_t i = 0; i < cu_window_seqlens.size() - 1; i++) {
        int32_t start = cu_window_seqlens[i];
        int32_t end = cu_window_seqlens[i + 1];
        for (int32_t h = start; h < end; h++) {
            for (int32_t w = start; w < end; w++) {
                window_mask[h * seq_len + w] = 0.0f;
            }
        }
    }
    
    // Cleanup
    clReleaseMemObject(window_mask_buf);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  OpenCL masks: " << duration.count() / 1000.0 << " ms" << std::endl;
}

} // namespace qwen_vl