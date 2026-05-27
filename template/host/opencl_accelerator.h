#pragma once

// ============================================================
// opencl_accelerator.h — OpenCL boilerplate wrapper
// Handles: platform init, context, command queue, buffer management
// ============================================================

#include <CL/cl.h>
#include <string>
#include <vector>

struct OpenCLContext {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    // Persistent buffers
    cl_mem src_buf;
    cl_mem norm_buf;
    cl_mem patches_buf;
    bool buffers_allocated;
    int src_w, src_h, src_stride;

    // Kernels
    cl_kernel resize_norm_kernel;
    cl_kernel transpose_kernel;
};

// Initialize OpenCL platform, device, context, queue
bool initOpenCL(OpenCLContext* ctx, std::string* error);

// Compile kernel.cl into program and create kernel objects
bool buildProgram(OpenCLContext* ctx, const char* kernel_path, std::string* error);

// Allocate or reuse persistent GPU buffers
bool allocateBuffers(OpenCLContext* ctx, int src_w, int src_h, int src_stride,
                     int dst_w, int dst_h, int seq_len, int input_dim,
                     std::string* error);

// Release all OpenCL resources
void cleanupOpenCL(OpenCLContext* ctx);

// Device info query helpers
int getComputeUnits(cl_device_id device);
size_t getMaxWorkGroupSize(cl_device_id device);
size_t getLocalMemSize(cl_device_id device);
std::string getDeviceName(cl_device_id device);
