#ifndef QWEN_VL_OPENCL_KERNELS_H
#define QWEN_VL_OPENCL_KERNELS_H

namespace qwen_vl {

const char* OPENCL_KERNEL_SOURCE = R"CLC(

__kernel void normalize_image(
    __global const uchar* src,
    __global float* dst,
    const float mean_r,
    const float mean_g,
    const float mean_b,
    const float std_r,
    const float std_g,
    const float std_b,
    const int height,
    const int width,
    const int channels
) {
    int h = get_global_id(0);
    int w = get_global_id(1);
    
    if (h >= height || w >= width) return;
    
    int base_idx = (h * width + w) * channels;
    
    float pixel_r = (float)src[base_idx + 0] / 255.0f;
    float pixel_g = (float)src[base_idx + 1] / 255.0f;
    float pixel_b = (float)src[base_idx + 2] / 255.0f;
    
    dst[base_idx + 0] = (pixel_r - mean_r) / std_r;
    dst[base_idx + 1] = (pixel_g - mean_g) / std_g;
    dst[base_idx + 2] = (pixel_b - mean_b) / std_b;
}

__kernel void transpose_to_patch(
    __global const float* input,
    __global float* output,
    const int T,
    const int H,
    const int W,
    const int C,
    const int patch_size,
    const int merge_size,
    const int temporal_patch_size,
    const int input_dim
) {
    int seqIdx = get_global_id(0);
    
    int gridT = T / temporal_patch_size;
    int gridH = H / patch_size;
    int gridW = W / patch_size;
    int merged_gridH = gridH / merge_size;
    int merged_gridW = gridW / merge_size;
    int seqLength = gridT * merged_gridH * merged_gridW * merge_size * merge_size;
    
    if (seqIdx >= seqLength) return;
    
    int tIdx = seqIdx / (merged_gridH * merged_gridW * merge_size * merge_size);
    int remainder = seqIdx % (merged_gridH * merged_gridW * merge_size * merge_size);
    int hIdx = remainder / (merged_gridW * merge_size * merge_size);
    remainder = remainder % (merged_gridW * merge_size * merge_size);
    int wIdx = remainder / (merge_size * merge_size);
    remainder = remainder % (merge_size * merge_size);
    int mergeH = remainder / merge_size;
    int mergeW = remainder % merge_size;
    
    for (int elemIdx = 0; elemIdx < input_dim; elemIdx++) {
        int cIdx = elemIdx / (temporal_patch_size * patch_size * patch_size);
        int temp = elemIdx % (temporal_patch_size * patch_size * patch_size);
        int tPatchIdx = temp / (patch_size * patch_size);
        temp = temp % (patch_size * patch_size);
        int patchH = temp / patch_size;
        int patchW = temp % patch_size;
        
        int srcT = tIdx * temporal_patch_size + tPatchIdx;
        int srcH = hIdx * merge_size * patch_size + mergeH * patch_size + patchH;
        int srcW = wIdx * merge_size * patch_size + mergeW * patch_size + patchW;
        int srcC = cIdx;
        
        int srcIdx = srcT * (H * W * C) + srcH * (W * C) + srcW * C + srcC;
        int dstIdx = seqIdx * input_dim + elemIdx;
        
        output[dstIdx] = input[srcIdx];
    }
}

__kernel void compute_rotary_emb(
    __global const int* grid_thw,
    __global float* pos_cos,
    __global float* pos_sin,
    const int seq_len,
    const int temporal_dim,
    const int height_dim,
    const int width_dim,
    const float rope_theta,
    const int merge_size,
    const int temporal_patch_size
) {
    int seq_idx = get_global_id(0);
    int dim_idx = get_global_id(1);
    
    int rope_dim = temporal_dim + height_dim + width_dim;
    
    if (seq_idx >= seq_len || dim_idx >= rope_dim) return;
    
    int T = grid_thw[0];
    int gridH = grid_thw[1];
    int gridW = grid_thw[2];
    
    // Simple grid position calculation (matches Python)
    int t = seq_idx / (gridH * gridW);
    int remainder = seq_idx % (gridH * gridW);
    int h = remainder / gridW;
    int w = remainder % gridW;
    
    int position;
    float inv_freq;
    
    if (dim_idx < height_dim) {
        // Height dimensions: [0-15]
        position = h;
        inv_freq = 1.0f / native_powr(rope_theta, (float)(2 * dim_idx) / (float)(2 * height_dim));
    }
    else if (dim_idx < height_dim + width_dim) {
        // Width dimensions: [16-31]
        position = w;
        int local_dim = dim_idx - height_dim;
        inv_freq = 1.0f / native_powr(rope_theta, (float)(2 * local_dim) / (float)(2 * width_dim));
    }
    else {
        // Temporal dimensions: [32-39]
        position = t;
        int local_dim = dim_idx - height_dim - width_dim;
        inv_freq = 1.0f / native_powr(rope_theta, (float)(2 * local_dim) / (float)(2 * temporal_dim));
    }
    
    float angle = (float)position * inv_freq;
    int output_idx = seq_idx * rope_dim + dim_idx;
    pos_cos[output_idx] = native_cos(angle);
    pos_sin[output_idx] = native_sin(angle);
}

__kernel void init_attention_masks(
    __global float* mask,
    const int seq_len
) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (i >= seq_len || j >= seq_len) return;
    
    // Initialize to -1000.0 (blocked attention)
    mask[i * seq_len + j] = -1000.0f;
}

)CLC";

} // namespace qwen_vl

#endif