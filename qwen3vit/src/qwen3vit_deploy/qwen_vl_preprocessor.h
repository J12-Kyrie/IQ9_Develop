#ifndef QWEN_VL_PREPROCESSOR_H
#define QWEN_VL_PREPROCESSOR_H

#include <string>
#include <vector>
#include <memory>

// Forward declarations to avoid including OpenCL headers here
struct _cl_platform_id;
struct _cl_device_id;
struct _cl_context;
struct _cl_command_queue;
struct _cl_program;
struct _cl_kernel;
typedef int32_t cl_int;
typedef _cl_platform_id* cl_platform_id;
typedef _cl_device_id* cl_device_id;
typedef _cl_context* cl_context;
typedef _cl_command_queue* cl_command_queue;
typedef _cl_program* cl_program;
typedef _cl_kernel* cl_kernel;

namespace qwen_vl {

struct QwenVLConfig {
    int32_t patch_size = 16;           // Qwen3-VL: 16 (was 14 in Qwen2.5-VL)
    int32_t temporal_patch_size = 2;
    int32_t merge_size = 2;
    int32_t min_image_tokens = 4;
    int32_t max_image_tokens = 16384;
    int32_t channels = 3;
    int32_t input_dim;                 // Computed: channels * temporal_patch_size * patch_size^2
    int32_t vit_pos_emb_dim = 32;      // Qwen3-VL: 32 (h+w only, was 40 in Qwen2.5-VL)
    std::vector<float> image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
    std::vector<float> image_std = {0.26862954f, 0.26130258f, 0.27577711f};
};

struct PreprocessedData {
    std::vector<float> pixel_values;
    std::vector<float> position_ids_cos;
    std::vector<float> position_ids_sin;
    // Note: attention masks are baked into the Qwen3-VL VEG binary; not needed as inputs
    std::vector<int32_t> image_grid_thw;
    int32_t seq_len = 0;
    int32_t resized_height = 0;
    int32_t resized_width = 0;
};

class OpenCLAccelerator; // Forward declaration

class QwenVLPreprocessor {
public:
    explicit QwenVLPreprocessor(const QwenVLConfig& config, bool use_opencl = true);
    ~QwenVLPreprocessor();
    
    std::pair<int32_t, int32_t> getResizedImageSize(int32_t height, int32_t width);
    
    PreprocessedData preprocessImage(const std::string& image_path, int32_t target_size = 0);
    PreprocessedData preprocessImage(const uint8_t* rgb_data, int32_t height, int32_t width,
                                    int32_t target_size = 0);
    
    void saveToRawFiles(const PreprocessedData& data, const std::string& output_dir,
                       bool create_input_list = true);
    
private:
    std::vector<uint8_t> loadImage(const std::string& path, int32_t& height,
                                  int32_t& width, int32_t& channels);
    std::vector<uint8_t> resizeImage(const uint8_t* src, int32_t src_h, int32_t src_w,
                                    int32_t dst_h, int32_t dst_w);
    
    void normalizeImageCPU(const uint8_t* src, float* dst, int32_t height,
                          int32_t width, int32_t channels);
    void transposeToPatchCPU(const float* normalized_image, float* output_patches,
                            int32_t T, int32_t H, int32_t W, int32_t C);
    void computeRotaryPosEmbCPU(const std::vector<int32_t>& grid_thw,
                               std::vector<float>& pos_cos,
                               std::vector<float>& pos_sin);
    
    QwenVLConfig config_;
    bool use_opencl_;
    std::unique_ptr<OpenCLAccelerator> opencl_;
};

} // namespace qwen_vl

#endif