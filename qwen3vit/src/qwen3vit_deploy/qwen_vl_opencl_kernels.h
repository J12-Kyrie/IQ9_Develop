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

__kernel void resize_bilinear(
    __global const uchar* src,
    __global uchar* dst,
    const int src_width,
    const int src_height,
    const int src_stride,
    const int dst_width,
    const int dst_height,
    const int channels
) {
    int dy = get_global_id(0);
    int dx = get_global_id(1);

    if (dy >= dst_height || dx >= dst_width) return;

    float sy = ((float)dy + 0.5f) * (float)src_height / (float)dst_height - 0.5f;
    float sx = ((float)dx + 0.5f) * (float)src_width / (float)dst_width - 0.5f;

    int x0 = (int)floor(sx);
    int y0 = (int)floor(sy);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    int idx00 = y0 * src_stride + x0 * channels;
    int idx01 = y0 * src_stride + x1 * channels;
    int idx10 = y1 * src_stride + x0 * channels;
    int idx11 = y1 * src_stride + x1 * channels;

    int dst_idx = (dy * dst_width + dx) * channels;

    for (int c = 0; c < channels; c++) {
        float v00 = (float)src[idx00 + c];
        float v01 = (float)src[idx01 + c];
        float v10 = (float)src[idx10 + c];
        float v11 = (float)src[idx11 + c];

        float val = v00 * (1.0f - fx) * (1.0f - fy)
                  + v01 * fx * (1.0f - fy)
                  + v10 * (1.0f - fx) * fy
                  + v11 * fx * fy;

        dst[dst_idx + c] = (uchar)clamp(val + 0.5f, 0.0f, 255.0f);
    }
}

// Fused resize + normalize: reads RGB24 uchar, bilinear resize, normalizes, outputs float32
// Eliminates intermediate uchar buffer and one host-device transfer
__kernel void resize_bilinear_normalized(
    __global const uchar* src,
    __global float* dst,
    const int src_width,
    const int src_height,
    const int src_stride,
    const int dst_width,
    const int dst_height,
    const float mean_r,
    const float mean_g,
    const float mean_b,
    const float inv_std_r,
    const float inv_std_g,
    const float inv_std_b
) {
    int dy = get_global_id(0);
    int dx = get_global_id(1);

    if (dy >= dst_height || dx >= dst_width) return;

    float sy = ((float)dy + 0.5f) * (float)src_height / (float)dst_height - 0.5f;
    float sx = ((float)dx + 0.5f) * (float)src_width / (float)dst_width - 0.5f;

    int x0 = (int)floor(sx);
    int y0 = (int)floor(sy);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);
    x0 = max(x0, 0);
    y0 = max(y0, 0);

    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    int idx00 = y0 * src_stride + x0 * 3;
    int idx01 = y0 * src_stride + x1 * 3;
    int idx10 = y1 * src_stride + x0 * 3;
    int idx11 = y1 * src_stride + x1 * 3;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w01 = fx * (1.0f - fy);
    float w10 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    int dst_idx = (dy * dst_width + dx) * 3;

    float r = (float)src[idx00] * w00 + (float)src[idx01] * w01
            + (float)src[idx10] * w10 + (float)src[idx11] * w11;
    float g = (float)src[idx00+1] * w00 + (float)src[idx01+1] * w01
            + (float)src[idx10+1] * w10 + (float)src[idx11+1] * w11;
    float b = (float)src[idx00+2] * w00 + (float)src[idx01+2] * w01
            + (float)src[idx10+2] * w10 + (float)src[idx11+2] * w11;

    dst[dst_idx]     = (r / 255.0f - mean_r) * inv_std_r;
    dst[dst_idx + 1] = (g / 255.0f - mean_g) * inv_std_g;
    dst[dst_idx + 2] = (b / 255.0f - mean_b) * inv_std_b;
}

// Fused transpose + normalize: reads uchar resized image, normalizes, writes patches
// Eliminates intermediate float normalized buffer
__kernel void transpose_to_patch_from_uchar(
    __global const uchar* resized,
    __global float* output,
    const int H,
    const int W,
    const int patch_size,
    const int merge_size,
    const int temporal_patch_size,
    const int input_dim,
    const float mean_r,
    const float mean_g,
    const float mean_b,
    const float inv_std_r,
    const float inv_std_g,
    const float inv_std_b
) {
    int seqIdx = get_global_id(0);

    int gridH = H / patch_size;
    int gridW = W / patch_size;
    int merged_gridH = gridH / merge_size;
    int merged_gridW = gridW / merge_size;
    int seqLength = merged_gridH * merged_gridW * merge_size * merge_size;

    if (seqIdx >= seqLength) return;

    int hIdx = seqIdx / (merged_gridW * merge_size * merge_size);
    int remainder = seqIdx % (merged_gridW * merge_size * merge_size);
    int wIdx = remainder / (merge_size * merge_size);
    remainder = remainder % (merge_size * merge_size);
    int mergeH = remainder / merge_size;
    int mergeW = remainder % merge_size;

    // Mean/std as float3 for vectorized access
    float3 mean = (float3)(mean_r, mean_g, mean_b);
    float3 inv_std = (float3)(inv_std_r, inv_std_g, inv_std_b);

    for (int elemIdx = 0; elemIdx < input_dim; elemIdx++) {
        int cIdx = elemIdx / (temporal_patch_size * patch_size * patch_size);
        int temp = elemIdx % (temporal_patch_size * patch_size * patch_size);
        int tPatchIdx = temp / (patch_size * patch_size);
        temp = temp % (patch_size * patch_size);
        int patchH = temp / patch_size;
        int patchW = temp % patch_size;

        int srcH = hIdx * merge_size * patch_size + mergeH * patch_size + patchH;
        int srcW = wIdx * merge_size * patch_size + mergeW * patch_size + patchW;
        int srcC = cIdx;

        int srcIdx = (srcH * W + srcW) * 3 + srcC;
        float val = (float)resized[srcIdx];

        // Normalize inline
        float mean_val = (srcC == 0) ? mean_r : (srcC == 1) ? mean_g : mean_b;
        float inv_std_val = (srcC == 0) ? inv_std_r : (srcC == 1) ? inv_std_g : inv_std_b;

        output[seqIdx * input_dim + elemIdx] = (val / 255.0f - mean_val) * inv_std_val;
    }
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