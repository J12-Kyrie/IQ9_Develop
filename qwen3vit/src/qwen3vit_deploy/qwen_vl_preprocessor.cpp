/*
 * Qwen VL Preprocessor Implementation
 * With OpenCL acceleration and CPU fallbacks
 */

#include "qwen_vl_preprocessor.h"
#include "qwen_vl_opencl_accelerator.h"
#include "qwen_vl_rope_tables.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <iostream>

// STB Image for image loading
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"

namespace qwen_vl {

// Helper functions
namespace {

inline int64_t roundByFactor(int64_t value, int64_t factor) {
    return static_cast<int64_t>(std::round(static_cast<double>(value) / factor) * factor);
}

inline int64_t floorByFactor(int64_t value, int64_t factor) {
    return static_cast<int64_t>(std::floor(static_cast<double>(value) / factor) * factor);
}

inline int64_t ceilByFactor(int64_t value, int64_t factor) {
    return static_cast<int64_t>(std::ceil(static_cast<double>(value) / factor) * factor);
}

} // anonymous namespace

QwenVLPreprocessor::QwenVLPreprocessor(const QwenVLConfig& config, bool use_opencl)
    : config_(config), use_opencl_(use_opencl) {
    
    // Validate config
    if (config_.patch_size <= 0 || config_.merge_size <= 0 || config_.temporal_patch_size <= 0) {
        throw std::runtime_error("Invalid config: patch_size, merge_size, and temporal_patch_size must be positive");
    }

    // Compute input_dim: channels * temporal_patch_size * patch_size^2
    // Qwen3-VL: 3 * 2 * 16 * 16 = 1536
    config_.input_dim = config_.channels * config_.temporal_patch_size *
                        config_.patch_size * config_.patch_size;

    std::cout << "Qwen3-VL config: patch_size=" << config_.patch_size
              << ", input_dim=" << config_.input_dim
              << ", rope_dim=" << config_.vit_pos_emb_dim << std::endl;

    // Initialize OpenCL if requested
    if (use_opencl_) {
        opencl_ = std::make_unique<OpenCLAccelerator>();
        opencl_->setConfig(&config_);
        if (!opencl_->initialize()) {
            std::cerr << "⚠️  OpenCL initialization failed, falling back to CPU" << std::endl;
            use_opencl_ = false;
            opencl_.reset();
        }
    }
}

QwenVLPreprocessor::~QwenVLPreprocessor() = default;

std::pair<int32_t, int32_t> QwenVLPreprocessor::getResizedImageSize(
    int32_t height, int32_t width) {

    int64_t const factor = config_.patch_size * config_.merge_size;
    int64_t const patchMergeProduct = config_.patch_size * config_.merge_size;
    int64_t const minPixels = config_.min_image_tokens * patchMergeProduct * patchMergeProduct;
    int64_t const maxPixels = config_.max_image_tokens * patchMergeProduct * patchMergeProduct;

    // For Qwen2.5-VL, aspect ratio constraint is handled differently
    // Remove this check or make it configurable
    // if (std::max(height, width) / std::min(height, width) > config_.max_ratio) {
    //     throw std::runtime_error("Aspect ratio must be smaller than " +
    //                            std::to_string(config_.max_ratio));
    // }

    int64_t hBar = std::max(factor, roundByFactor(height, factor));
    int64_t wBar = std::max(factor, roundByFactor(width, factor));

    if (hBar * wBar > maxPixels) {
        double beta = std::sqrt(static_cast<double>(height * width) / maxPixels);
        hBar = floorByFactor(static_cast<int64_t>(height / beta), factor);
        wBar = floorByFactor(static_cast<int64_t>(width / beta), factor);
    } else if (hBar * wBar < minPixels) {
        double beta = std::sqrt(static_cast<double>(minPixels) / (height * width));
        hBar = ceilByFactor(static_cast<int64_t>(height * beta), factor);
        wBar = ceilByFactor(static_cast<int64_t>(width * beta), factor);
    }

    return {static_cast<int32_t>(hBar), static_cast<int32_t>(wBar)};
}

std::vector<uint8_t> QwenVLPreprocessor::loadImage(
    const std::string& path, int32_t& height, int32_t& width, int32_t& channels) {

    int h, w, c;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &c, 3); // Force RGB

    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    height = h;
    width = w;
    channels = 3;

    std::vector<uint8_t> result(data, data + (h * w * 3));
    stbi_image_free(data);

    return result;
}

std::vector<uint8_t> QwenVLPreprocessor::resizeImage(
    const uint8_t* src, int32_t src_h, int32_t src_w,
    int32_t dst_h, int32_t dst_w) {

    std::vector<uint8_t> dst(dst_h * dst_w * 3);

    // Use STB image resize with cubic filter to match Python's PIL.Image.BICUBIC (resample=3)
    stbir_resize_uint8_srgb(
        src, src_w, src_h, 0,
        dst.data(), dst_w, dst_h, 0,
        STBIR_RGB
    );

    return dst;
}

void QwenVLPreprocessor::normalizeImageCPU(
    const uint8_t* src, float* dst, int32_t height, int32_t width, int32_t channels) {

    // Normalize: (pixel / 255.0 - mean) / std
    for (int32_t h = 0; h < height; ++h) {
        for (int32_t w = 0; w < width; ++w) {
            for (int32_t c = 0; c < channels; ++c) {
                int32_t idx = h * width * channels + w * channels + c;
                float pixel = static_cast<float>(src[idx]) / 255.0f;
                dst[idx] = (pixel - config_.image_mean[c]) / config_.image_std[c];
            }
        }
    }
}

void QwenVLPreprocessor::transposeToPatchCPU(
    const float* normalized_image, float* output_patches,
    int32_t T, int32_t H, int32_t W, int32_t C) {

    int32_t const gridT = T / config_.temporal_patch_size;
    int32_t const gridH = H / config_.patch_size;
    int32_t const gridW = W / config_.patch_size;
    int32_t const merged_gridH = gridH / config_.merge_size;
    int32_t const merged_gridW = gridW / config_.merge_size;
    int32_t const seqLength = gridT * merged_gridH * merged_gridW * 
                             config_.merge_size * config_.merge_size;
    int32_t const inputDim = config_.input_dim;

    // Process each sequence position
    for (int32_t seqIdx = 0; seqIdx < seqLength; ++seqIdx) {
        // Calculate sequence coordinates
        int32_t const tIdx = seqIdx / (merged_gridH * merged_gridW * 
                                      config_.merge_size * config_.merge_size);
        int32_t remainder = seqIdx % (merged_gridH * merged_gridW * 
                                      config_.merge_size * config_.merge_size);
        int32_t const hIdx = remainder / (merged_gridW * config_.merge_size * config_.merge_size);
        remainder = remainder % (merged_gridW * config_.merge_size * config_.merge_size);
        int32_t const wIdx = remainder / (config_.merge_size * config_.merge_size);
        remainder = remainder % (config_.merge_size * config_.merge_size);
        int32_t const mergeH = remainder / config_.merge_size;
        int32_t const mergeW = remainder % config_.merge_size;

        // Process each element in this sequence position
        for (int32_t elemIdx = 0; elemIdx < inputDim; ++elemIdx) {
            // Calculate coordinates within the patch
            int32_t const cIdx = elemIdx / (config_.temporal_patch_size * 
                                           config_.patch_size * config_.patch_size);
            int32_t temp = elemIdx % (config_.temporal_patch_size * 
                                      config_.patch_size * config_.patch_size);
            int32_t const tPatchIdx = temp / (config_.patch_size * config_.patch_size);
            temp = temp % (config_.patch_size * config_.patch_size);
            int32_t const patchH = temp / config_.patch_size;
            int32_t const patchW = temp % config_.patch_size;

            // Calculate source coordinates
            int32_t const srcT = tIdx * config_.temporal_patch_size + tPatchIdx;
            int32_t const srcH = hIdx * config_.merge_size * config_.patch_size + 
                                mergeH * config_.patch_size + patchH;
            int32_t const srcW = wIdx * config_.merge_size * config_.patch_size + 
                                mergeW * config_.patch_size + patchW;
            int32_t const srcC = cIdx;

            // Calculate indices
            int32_t const srcIdx = srcT * H * W * C + srcH * W * C + srcW * C + srcC;
            int32_t const dstIdx = seqIdx * inputDim + elemIdx;

            output_patches[dstIdx] = normalized_image[srcIdx];
        }
    }
}

void QwenVLPreprocessor::computeRotaryPosEmbCPU(
    const std::vector<int32_t>& grid_thw,
    std::vector<float>& pos_cos,
    std::vector<float>& pos_sin) {

    // Qwen3-VL RoPE: h+w only, block+intra patch ordering, no temporal axis.
    // Matches Qwen3VLVisionEncoder.rot_pos_emb() exactly.
    //
    // Algorithm:
    //   inv_freq[j] = 1 / (10000 ^ (2j / rope_dim))  for j=0..half_dim-1
    //   freq_table[pos] = pos * inv_freq              shape [max_hw, half_dim]
    //   Patches ordered: for each 2x2 block (block_row, block_col),
    //     then for each (intra_row, intra_col) within the block:
    //       embeddings[:half_dim] = freq_table[h_pos]
    //       embeddings[half_dim:] = freq_table[w_pos]
    //   pos_cos = cos(embeddings), pos_sin = sin(embeddings)

    const int32_t rope_dim = config_.vit_pos_emb_dim;  // 32
    const int32_t half_dim = rope_dim / 2;              // 16
    const float theta = 10000.0f;

    // T is always 1 for still images; gridH, gridW are the patch grid (e.g., 28x28)
    int32_t gridH = grid_thw[1];
    int32_t gridW = grid_thw[2];
    int32_t seq_len = gridH * gridW;  // 784 for 448x448

    pos_cos.resize(seq_len * rope_dim);
    pos_sin.resize(seq_len * rope_dim);

    // Precompute inv_freq and freq_table for positions 0..max_hw-1
    int32_t max_hw = std::max(gridH, gridW);
    std::vector<float> inv_freq(half_dim);
    for (int32_t j = 0; j < half_dim; ++j) {
        inv_freq[j] = 1.0f / std::pow(theta, (float)(2 * j) / (float)rope_dim);
    }
    // freq_table[pos * half_dim + j] = pos * inv_freq[j]
    std::vector<float> freq_table(max_hw * half_dim);
    for (int32_t pos = 0; pos < max_hw; ++pos) {
        for (int32_t j = 0; j < half_dim; ++j) {
            freq_table[pos * half_dim + j] = (float)pos * inv_freq[j];
        }
    }

    // Generate patches in block+intra order (matching Python rot_pos_emb)
    int32_t merged_h = gridH / config_.merge_size;
    int32_t merged_w = gridW / config_.merge_size;
    int32_t patch_idx = 0;

    for (int32_t br = 0; br < merged_h; ++br) {
        for (int32_t bc = 0; bc < merged_w; ++bc) {
            for (int32_t ir = 0; ir < config_.merge_size; ++ir) {
                for (int32_t ic = 0; ic < config_.merge_size; ++ic) {
                    int32_t h_pos = br * config_.merge_size + ir;
                    int32_t w_pos = bc * config_.merge_size + ic;

                    // First half_dim: h embedding; second half_dim: w embedding
                    const float* h_freqs = &freq_table[h_pos * half_dim];
                    const float* w_freqs = &freq_table[w_pos * half_dim];
                    float* out_cos = &pos_cos[patch_idx * rope_dim];
                    float* out_sin = &pos_sin[patch_idx * rope_dim];

                    for (int32_t j = 0; j < half_dim; ++j) {
                        out_cos[j]            = std::cos(h_freqs[j]);
                        out_sin[j]            = std::sin(h_freqs[j]);
                        out_cos[half_dim + j] = std::cos(w_freqs[j]);
                        out_sin[half_dim + j] = std::sin(w_freqs[j]);
                    }
                    ++patch_idx;
                }
            }
        }
    }
}


PreprocessedData QwenVLPreprocessor::preprocessImage(
    const std::string& image_path, int32_t target_size) {

    // Load image
    int32_t orig_h, orig_w, channels;
    auto image_data = loadImage(image_path, orig_h, orig_w, channels);

    return preprocessImage(image_data.data(), orig_h, orig_w, target_size);
}

PreprocessedData QwenVLPreprocessor::preprocessImage(
    const uint8_t* rgb_data, int32_t height, int32_t width, int32_t target_size) {

    std::cout << "\n=== Preprocessing Image ===" << std::endl;
    PreprocessedData result;

    // Get resized dimensions
    auto [resized_h, resized_w] = getResizedImageSize(height, width);

    // Override with target_size if specified
    if (target_size > 0) {
        int32_t factor = config_.patch_size * config_.merge_size;
        resized_h = resized_w = (target_size / factor) * factor;
    }

    result.resized_height = resized_h;
    result.resized_width = resized_w;
    std::cout << "Target size: " << resized_h << "x" << resized_w << std::endl;

    // Resize image
    std::vector<uint8_t> resized;
    if (height != resized_h || width != resized_w) {
        resized = resizeImage(rgb_data, height, width, resized_h, resized_w);
    } else {
        resized.assign(rgb_data, rgb_data + (height * width * 3));
    }

    // Normalize image [T, H, W, C]
    int32_t T = config_.temporal_patch_size;
    std::vector<float> normalized(T * resized_h * resized_w * 3);

    // Replicate image T times (for temporal dimension)
    for (int32_t t = 0; t < T; ++t) {
        if (use_opencl_ && opencl_) {
            opencl_->normalizeImage(resized.data(),
                                   normalized.data() + t * resized_h * resized_w * 3,
                                   resized_h, resized_w, 3,
                                   config_.image_mean.data(), config_.image_std.data());
        } else {
            normalizeImageCPU(resized.data(),
                            normalized.data() + t * resized_h * resized_w * 3,
                            resized_h, resized_w, 3);
        }
    }

    // Compute grid dimensions
    int32_t gridT = T / config_.temporal_patch_size;
    int32_t gridH = resized_h / config_.patch_size;
    int32_t gridW = resized_w / config_.patch_size;
    int32_t merged_gridH = gridH / config_.merge_size;
    int32_t merged_gridW = gridW / config_.merge_size;
    result.seq_len = gridT * merged_gridH * merged_gridW * 
                    config_.merge_size * config_.merge_size;

    result.image_grid_thw = {gridT, gridH, gridW};

    std::cout << "Grid dimensions: [" << gridT << ", " << gridH << ", " << gridW << "]" << std::endl;
    std::cout << "Sequence length: " << result.seq_len << std::endl;

    // Validate sequence length
    if (result.seq_len <= 0 || result.seq_len > 10000) {
        throw std::runtime_error("Invalid sequence length: " + std::to_string(result.seq_len));
    }

    // Transpose to patch format [seq_len, input_dim]
    result.pixel_values.resize(result.seq_len * config_.input_dim);
    
    if (use_opencl_ && opencl_) {
        opencl_->transposeToPatch(normalized.data(), result.pixel_values.data(),
                                 T, resized_h, resized_w, 3,
                                 config_.patch_size, config_.merge_size,
                                 config_.temporal_patch_size, config_.input_dim);
    } else {
        transposeToPatchCPU(normalized.data(), result.pixel_values.data(),
                           T, resized_h, resized_w, 3);
    }

    // Compute rotary position embeddings
    // The correct sequence length for position embeddings is gridT * gridH * gridW
    int32_t pos_seq_len = gridT * gridH * gridW;
    result.position_ids_cos.resize(pos_seq_len * config_.vit_pos_emb_dim);
    result.position_ids_sin.resize(pos_seq_len * config_.vit_pos_emb_dim);

    // Use precomputed RoPE lookup tables for 448x448 images (instead of computing)
    // This ensures 100% match with Python and is faster
    if (pos_seq_len == ROPE_SEQ_LEN_448 && config_.vit_pos_emb_dim == ROPE_DIM) {
        // Copy from lookup tables
        std::memcpy(result.position_ids_cos.data(), ROPE_COS_TABLE_448,
                    pos_seq_len * config_.vit_pos_emb_dim * sizeof(float));
        std::memcpy(result.position_ids_sin.data(), ROPE_SIN_TABLE_448,
                    pos_seq_len * config_.vit_pos_emb_dim * sizeof(float));
        std::cout << "  Using precomputed RoPE tables (448x448)" << std::endl;
    } else {
        // Fallback to computation for other sizes (should not happen with 448x448 restriction)
        std::cerr << "⚠️  WARNING: Image size not 448x448! RoPE computation not supported." << std::endl;
        std::cerr << "    Expected seq_len=" << ROPE_SEQ_LEN_448 << ", got " << pos_seq_len << std::endl;
        throw std::runtime_error("Only 448x448 images are supported");
    }

    // Attention masks are baked into the Qwen3-VL VEG binary (all-zero, full attention)
    // and are not passed as runtime inputs.

    std::cout << "✅ Preprocessing complete!" << std::endl;
    return result;
}

void QwenVLPreprocessor::saveToRawFiles(
    const PreprocessedData& data, const std::string& output_dir,
    bool create_input_list) {

    // Helper to save tensor
    auto saveTensor = [&](const std::vector<float>& tensor, const std::string& name,
                         const std::vector<int32_t>& shape) {
        std::string filepath = output_dir + "/" + name + ".raw";
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }
        ofs.write(reinterpret_cast<const char*>(tensor.data()),
                 tensor.size() * sizeof(float));
        ofs.close();

        printf("Saved %s | shape=[", filepath.c_str());
        for (size_t i = 0; i < shape.size(); ++i) {
            printf("%d%s", shape[i], i < shape.size() - 1 ? "," : "");
        }
        printf("] | size=%zu bytes\n", tensor.size() * sizeof(float));
    };

    // Save all tensors (Qwen3-VL VEG has 3 inputs: pixel_values, cos, sin)
    saveTensor(data.pixel_values, "pixel_values",
              {data.seq_len, 1, 1, config_.input_dim});
    saveTensor(data.position_ids_cos, "position_ids_cos",
              {data.seq_len, config_.vit_pos_emb_dim});
    saveTensor(data.position_ids_sin, "position_ids_sin",
              {data.seq_len, config_.vit_pos_emb_dim});

    // Create input_list.txt
    if (create_input_list) {
        std::string input_list_path = output_dir + "/input_list.txt";
        std::ofstream ofs(input_list_path);
        if (!ofs) {
            throw std::runtime_error("Failed to create input_list.txt");
        }

        ofs << "pixel_values:=" << output_dir << "/pixel_values.raw "
            << "position_ids_cos:=" << output_dir << "/position_ids_cos.raw "
            << "position_ids_sin:=" << output_dir << "/position_ids_sin.raw";

        ofs.close();
        printf("Created %s\n", input_list_path.c_str());
    }

    printf("\n✅ SUCCESS: Preprocessed inputs saved for seq_len=%d\n", data.seq_len);
}

} // namespace qwen_vl