/// @file FaceProcessor.hpp
/// @brief 人脸检测+识别编排类: SCRFD + ArcFace 全链路串联
///
/// 输入: NV12 帧 cl_mem + FramePlaneInfo
/// 输出: vector<FaceResult> (bbox + landmarks + 512-dim embedding)
///
/// 内部链路 (所有子模块均已板上验证通过):
///   NV12 → FacePreprocess::RunScrfd()      — GPU letterbox 1.19ms
///        → QnnInferencer::Execute(SCRFD)   — HTP 推理 10.36ms
///        → ScrfdDecode() + NMS             — CPU 解码
///        → 每个人脸:
///            ComputeInverseAffine()        — Umeyama 逆仿射
///            FacePreprocess::RunArcface()  — GPU 仿射裁剪 0.107ms
///            ArcFaceInfer::Infer()         — HTP 推理 6.47ms + L2 归一化
///        → vector<FaceResult>

#pragma once

#include "face_preprocess/FacePreprocess.hpp"
#include "face_preprocess/inverse_affine.hpp"
#include "scrfd_infer/FaceTypes.hpp"
#include "scrfd_infer/QnnInferencer.hpp"
#include "arcface_infer/ArcFaceInfer.hpp"
#include "mem_management/opencl_loader.hpp"
#include "mem_management/mem_types.hpp"

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace face_infer {

/// FaceProcessor 初始化配置
struct FaceProcessorConfig {
    std::string qnn_backend   = "/usr/lib/libQnnHtp.so";
    std::string qnn_system    = "/usr/lib/libQnnSystem.so";
    std::string scrfd_model;                 // 必填: SCRFD .bin
    std::string arcface_model;               // 可选: 空 = 仅检测
    std::string opencl_kernel_path;          // 必填: .cl kernel 目录
    float conf_threshold = 0.5f;
    float nms_threshold  = 0.4f;
    int   max_faces      = kFaceMaxDetections;
};

class FaceProcessor {
public:
    FaceProcessor() = default;
    ~FaceProcessor();
    FaceProcessor(const FaceProcessor&) = delete;
    FaceProcessor& operator=(const FaceProcessor&) = delete;

    /// 初始化: 加载 SCRFD + ArcFace 模型, 初始化 OpenCL
    bool Init(const FaceProcessorConfig& config);

    /// 处理单帧: 已导入 OpenCL 的 NV12 cl_mem → 人脸检测 + 识别
    std::vector<FaceResult> ProcessFrame(cl_mem nv12_cl,
                                         const FramePlaneInfo& plane);

    /// 余弦相似度 (两个 L2 归一化嵌入向量)
    static float Similarity(const float* emb1, const float* emb2);

    void Destroy();

    /// 获取 OpenCL context (供外部 DmaBuffer::BindOpenCl 使用)
    cl_context GetContext() const;

    /// 是否启用 ArcFace 识别
    bool HasArcFace() const { return arcface_enabled_; }

private:
    /// Phase A: SCRFD 全帧检测
    std::vector<FaceDetection> DetectFaces(cl_mem nv12_cl,
                                            const FramePlaneInfo& plane);

    /// Phase B: 单人脸 ArcFace 嵌入提取
    bool ExtractEmbedding(cl_mem nv12_cl,
                          const FramePlaneInfo& plane,
                          const FaceDetection& det,
                          float out_embedding[kFaceEmbeddingDim]);

    std::shared_ptr<OpenClLoader>   ocl_;
    FacePreprocess                  gpu_;
    QnnInferencer                   scrfd_qnn_;
    arcface_infer::ArcFaceInfer     arcface_;

    FaceProcessorConfig config_;
    bool arcface_enabled_ = false;
    bool initialized_     = false;

};

}  // namespace face_infer
