//==============================================================================
//
//  VEG Combined Runner Application
//  Combines OpenCL image preprocessing with QNN vision encoder inference
//  Takes image as input, outputs VEG encoding directly (no intermediate files)
//
//==============================================================================

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <cstring>
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <chrono>

// QNN Headers
#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/QnnCommon.h"
#include "QNN/QnnDevice.h"
#include "QNN/QnnContext.h"
#include "QNN/System/QnnSystemInterface.h"
#include "QNN/System/QnnSystemContext.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpContext.h"

// OpenCL Preprocessor
#include "qwen_vl_preprocessor.h"

#define LOG_INFO(fmt, ...) fprintf(stdout, "[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) fprintf(stdout, "[WARN] " fmt "\n", ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) fprintf(stdout, "[DEBUG] " fmt "\n", ##__VA_ARGS__)

// ============================================================================
// QNN Tensor accessor/modifier macros (from SampleApp QnnTypeMacros.hpp)
// ============================================================================

inline uint32_t getQnnTensorId(const Qnn_Tensor_t& tensor) { return tensor.v1.id; }
inline const char* getQnnTensorName(const Qnn_Tensor_t& tensor) { return tensor.v1.name; }
inline Qnn_TensorType_t getQnnTensorType(const Qnn_Tensor_t& tensor) { return tensor.v1.type; }
inline Qnn_TensorDataFormat_t getQnnTensorDataFormat(const Qnn_Tensor_t& tensor) { return tensor.v1.dataFormat; }
inline Qnn_DataType_t getQnnTensorDataType(const Qnn_Tensor_t& tensor) { return tensor.v1.dataType; }
inline Qnn_QuantizeParams_t getQnnTensorQuantParams(const Qnn_Tensor_t& tensor) { return tensor.v1.quantizeParams; }
inline uint32_t getQnnTensorRank(const Qnn_Tensor_t& tensor) { return tensor.v1.rank; }
inline uint32_t* getQnnTensorDimensions(const Qnn_Tensor_t& tensor) { return tensor.v1.dimensions; }
inline Qnn_TensorMemType_t getQnnTensorMemType(const Qnn_Tensor_t& tensor) { return tensor.v1.memType; }
inline Qnn_ClientBuffer_t getQnnTensorClientBuf(const Qnn_Tensor_t& tensor) { return tensor.v1.clientBuf; }

inline uint8_t* getQnnTensorIsDynamicDimensions(const Qnn_Tensor_t& tensor) {
    return (tensor.version == QNN_TENSOR_VERSION_2) ? tensor.v2.isDynamicDimensions : nullptr;
}
inline Qnn_SparseParams_t getQnnTensorSparseParams(const Qnn_Tensor_t& tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_2) {
        return tensor.v2.sparseParams;
    }
    Qnn_SparseParams_t defaultParams = QNN_SPARSE_PARAMS_INIT;
    return defaultParams;
}

// Pointer versions
inline uint32_t getQnnTensorId(const Qnn_Tensor_t* tensor) { return getQnnTensorId(*tensor); }
inline const char* getQnnTensorName(const Qnn_Tensor_t* tensor) { return getQnnTensorName(*tensor); }
inline Qnn_TensorType_t getQnnTensorType(const Qnn_Tensor_t* tensor) { return getQnnTensorType(*tensor); }
inline Qnn_TensorDataFormat_t getQnnTensorDataFormat(const Qnn_Tensor_t* tensor) { return getQnnTensorDataFormat(*tensor); }
inline Qnn_DataType_t getQnnTensorDataType(const Qnn_Tensor_t* tensor) { return getQnnTensorDataType(*tensor); }
inline Qnn_QuantizeParams_t getQnnTensorQuantParams(const Qnn_Tensor_t* tensor) {
    if (tensor) {
        return getQnnTensorQuantParams(*tensor);
    }
    Qnn_QuantizeParams_t defaultParams = QNN_QUANTIZE_PARAMS_INIT;
    return defaultParams;
}
inline uint32_t getQnnTensorRank(const Qnn_Tensor_t* tensor) { return tensor ? getQnnTensorRank(*tensor) : 0u; }
inline uint32_t* getQnnTensorDimensions(const Qnn_Tensor_t* tensor) { return getQnnTensorDimensions(*tensor); }
inline Qnn_TensorMemType_t getQnnTensorMemType(const Qnn_Tensor_t* tensor) { return getQnnTensorMemType(*tensor); }
inline Qnn_ClientBuffer_t getQnnTensorClientBuf(const Qnn_Tensor_t* tensor) { return getQnnTensorClientBuf(*tensor); }

// Setters
inline void setQnnTensorId(Qnn_Tensor_t& tensor, uint32_t id) { tensor.v1.id = id; }
inline void setQnnTensorName(Qnn_Tensor_t& tensor, const char* name) { tensor.v1.name = name; }
inline void setQnnTensorType(Qnn_Tensor_t& tensor, Qnn_TensorType_t type) { tensor.v1.type = type; }
inline void setQnnTensorDataFormat(Qnn_Tensor_t& tensor, Qnn_TensorDataFormat_t format) { tensor.v1.dataFormat = format; }
inline void setQnnTensorDataType(Qnn_Tensor_t& tensor, Qnn_DataType_t dataType) { tensor.v1.dataType = dataType; }
inline void setQnnTensorQuantParams(Qnn_Tensor_t& tensor, Qnn_QuantizeParams_t params) { tensor.v1.quantizeParams = params; }
inline void setQnnTensorRank(Qnn_Tensor_t& tensor, uint32_t rank) { tensor.v1.rank = rank; }
inline void setQnnTensorDimensions(Qnn_Tensor_t& tensor, uint32_t* dims) { tensor.v1.dimensions = dims; }
inline void setQnnTensorMemType(Qnn_Tensor_t& tensor, Qnn_TensorMemType_t memType) { tensor.v1.memType = memType; }
inline void setQnnTensorClientBuf(Qnn_Tensor_t& tensor, Qnn_ClientBuffer_t clientBuf) { tensor.v1.clientBuf = clientBuf; }

// Pointer versions of setters
inline void setQnnTensorId(Qnn_Tensor_t* tensor, uint32_t id) { setQnnTensorId(*tensor, id); }
inline void setQnnTensorName(Qnn_Tensor_t* tensor, const char* name) { setQnnTensorName(*tensor, name); }
inline void setQnnTensorType(Qnn_Tensor_t* tensor, Qnn_TensorType_t type) { setQnnTensorType(*tensor, type); }
inline void setQnnTensorDataFormat(Qnn_Tensor_t* tensor, Qnn_TensorDataFormat_t format) { setQnnTensorDataFormat(*tensor, format); }
inline void setQnnTensorDataType(Qnn_Tensor_t* tensor, Qnn_DataType_t dataType) { setQnnTensorDataType(*tensor, dataType); }
inline void setQnnTensorQuantParams(Qnn_Tensor_t* tensor, Qnn_QuantizeParams_t params) { setQnnTensorQuantParams(*tensor, params); }
inline void setQnnTensorRank(Qnn_Tensor_t* tensor, uint32_t rank) { setQnnTensorRank(*tensor, rank); }
inline void setQnnTensorDimensions(Qnn_Tensor_t* tensor, uint32_t* dims) { setQnnTensorDimensions(*tensor, dims); }
inline void setQnnTensorMemType(Qnn_Tensor_t* tensor, Qnn_TensorMemType_t memType) { setQnnTensorMemType(*tensor, memType); }
inline void setQnnTensorClientBuf(Qnn_Tensor_t* tensor, Qnn_ClientBuffer_t clientBuf) { setQnnTensorClientBuf(*tensor, clientBuf); }

// Macros for convenient access
#define QNN_TENSOR_GET_ID(tensor)                    getQnnTensorId(tensor)
#define QNN_TENSOR_GET_NAME(tensor)                  getQnnTensorName(tensor)
#define QNN_TENSOR_GET_TYPE(tensor)                  getQnnTensorType(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor)           getQnnTensorDataFormat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor)             getQnnTensorDataType(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor)          getQnnTensorQuantParams(tensor)
#define QNN_TENSOR_GET_RANK(tensor)                  getQnnTensorRank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor)            getQnnTensorDimensions(tensor)
#define QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensor) getQnnTensorIsDynamicDimensions(tensor)
#define QNN_TENSOR_GET_SPARSE_PARAMS(tensor)         getQnnTensorSparseParams(tensor)
#define QNN_TENSOR_GET_MEM_TYPE(tensor)              getQnnTensorMemType(tensor)
#define QNN_TENSOR_GET_CLIENT_BUF(tensor)            getQnnTensorClientBuf(tensor)

#define QNN_TENSOR_SET_ID(tensor, value)             setQnnTensorId(tensor, value)
#define QNN_TENSOR_SET_NAME(tensor, value)           setQnnTensorName(tensor, value)
#define QNN_TENSOR_SET_TYPE(tensor, value)           setQnnTensorType(tensor, value)
#define QNN_TENSOR_SET_DATA_FORMAT(tensor, value)    setQnnTensorDataFormat(tensor, value)
#define QNN_TENSOR_SET_DATA_TYPE(tensor, value)      setQnnTensorDataType(tensor, value)
#define QNN_TENSOR_SET_QUANT_PARAMS(tensor, value)   setQnnTensorQuantParams(tensor, value)
#define QNN_TENSOR_SET_RANK(tensor, value)           setQnnTensorRank(tensor, value)
#define QNN_TENSOR_SET_DIMENSIONS(tensor, value)     setQnnTensorDimensions(tensor, value)
#define QNN_TENSOR_SET_MEM_TYPE(tensor, value)       setQnnTensorMemType(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF(tensor, value)     setQnnTensorClientBuf(tensor, value)

// Function pointer types
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                          uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
    const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

// Graph info structure
struct GraphInfo_t {
    Qnn_GraphHandle_t graph;
    char* graphName;
    Qnn_Tensor_t* inputTensors;
    uint32_t numInputTensors;
    Qnn_Tensor_t* outputTensors;
    uint32_t numOutputTensors;
};

// Helper function to deep copy tensor info
static bool deepCopyQnnTensorInfo(Qnn_Tensor_t* dst, const Qnn_Tensor_t* src) {
    if (nullptr == dst || nullptr == src) {
        LOG_ERROR("deepCopyQnnTensorInfo: received nullptr");
        return false;
    }

    dst->version = src->version;

    const char* tensorName = QNN_TENSOR_GET_NAME(src);
    if (!tensorName) {
        QNN_TENSOR_SET_NAME(dst, nullptr);
    } else {
        QNN_TENSOR_SET_NAME(dst, strdup(tensorName));
    }

    QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
    QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
    QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
    QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));

    Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
    qParams.encodingDefinition = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
    qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;

    if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qParams.scaleOffsetEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
    } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qParams.axisScaleOffsetEncoding.axis = QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
        qParams.axisScaleOffsetEncoding.numScaleOffsets = QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
        if (QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets > 0) {
            qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t*)malloc(
                QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets * sizeof(Qnn_ScaleOffset_t));
            if (qParams.axisScaleOffsetEncoding.scaleOffset) {
                for (size_t idx = 0; idx < QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets; idx++) {
                    qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
                        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].scale;
                    qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
                        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].offset;
                }
            }
        }
    }
    QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);

    QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
    QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);

    if (QNN_TENSOR_GET_RANK(src) > 0) {
        QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t*)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
        if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
            memcpy(QNN_TENSOR_GET_DIMENSIONS(dst), QNN_TENSOR_GET_DIMENSIONS(src),
                   QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
        }
    }

    return true;
}

// Quantize FP32 to uint16
static uint16_t quantizeToUint16(float value, float scale, int32_t offset) {
    const double trueBitWidthMax = 65535.0;
    double encodingMin = static_cast<double>(offset) * static_cast<double>(scale);
    double encodingMax = (trueBitWidthMax + static_cast<double>(offset)) * static_cast<double>(scale);
    double encodingRange = encodingMax - encodingMin;

    int quantizedValue = static_cast<int>(round(trueBitWidthMax * (value - encodingMin) / encodingRange));

    if (quantizedValue < 0) quantizedValue = 0;
    if (quantizedValue > 65535) quantizedValue = 65535;

    return static_cast<uint16_t>(quantizedValue);
}

// Dequantize uint16 to FP32
static float dequantizeFromUint16(uint16_t value, float scale, int32_t offset) {
    double quantizedValue = static_cast<double>(value);
    double offsetDouble = static_cast<double>(offset);
    return static_cast<float>((quantizedValue + offsetDouble) * static_cast<double>(scale));
}

// Float32 to Float16 conversion
static uint16_t float32ToFloat16(float value) {
    uint32_t f32;
    memcpy(&f32, &value, sizeof(float));

    uint32_t sign = (f32 >> 31) & 0x1;
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x7FFFFF;

    uint16_t f16;

    if (exponent > 15) {
        f16 = (sign << 15) | (0x1F << 10);
    } else if (exponent < -14) {
        if (exponent < -24) {
            f16 = (sign << 15);
        } else {
            mantissa |= 0x800000;
            int shift = -14 - exponent;
            mantissa >>= shift;
            f16 = (sign << 15) | (mantissa >> 13);
        }
    } else {
        uint16_t biasedExp = exponent + 15;
        f16 = (sign << 15) | (biasedExp << 10) | (mantissa >> 13);
    }

    return f16;
}

// Float16 to Float32 conversion
static float float16ToFloat32(uint16_t value) {
    uint32_t sign = (value >> 15) & 0x1;
    uint32_t exponent = (value >> 10) & 0x1F;
    uint32_t mantissa = value & 0x3FF;

    uint32_t f32;

    if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign << 31;
        } else {
            exponent = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        f32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        f32 = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

// Helper to calculate buffer size
static size_t calculateTensorSize(const Qnn_Tensor_t* tensor) {
    if (!tensor) return 0;

    uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
    uint32_t* dims = QNN_TENSOR_GET_DIMENSIONS(tensor);

    if (rank == 0 || !dims) return 0;

    size_t elementCount = 1;
    for (uint32_t i = 0; i < rank; i++) {
        elementCount *= dims[i];
    }

    size_t elementSize = 1;
    Qnn_DataType_t dataType = QNN_TENSOR_GET_DATA_TYPE(tensor);

    switch (dataType) {
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_BOOL_8:
            elementSize = 1;
            break;
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_FLOAT_16:
            elementSize = 2;
            break;
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_SFIXED_POINT_32:
        case QNN_DATATYPE_UFIXED_POINT_32:
            elementSize = 4;
            break;
        case QNN_DATATYPE_INT_64:
        case QNN_DATATYPE_UINT_64:
        case QNN_DATATYPE_FLOAT_64:
            elementSize = 8;
            break;
        default:
            elementSize = 1;
            break;
    }

    return elementCount * elementSize;
}

// Helper to copy tensors info
static bool copyTensorsInfo(const Qnn_Tensor_t* tensorsInfoSrc, Qnn_Tensor_t*& tensorWrappers, uint32_t tensorsCount) {
    tensorWrappers = (Qnn_Tensor_t*)calloc(tensorsCount, sizeof(Qnn_Tensor_t));
    if (nullptr == tensorWrappers) {
        LOG_ERROR("Failed to allocate memory for tensorWrappers");
        return false;
    }

    for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
        tensorWrappers[tIdx] = QNN_TENSOR_INIT;
        deepCopyQnnTensorInfo(&tensorWrappers[tIdx], &tensorsInfoSrc[tIdx]);
    }

    return true;
}

// Helper to copy graph info V1
static bool copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc, GraphInfo_t* graphInfoDst) {
    graphInfoDst->graphName = nullptr;
    if (graphInfoSrc->graphName) {
        graphInfoDst->graphName = strdup(graphInfoSrc->graphName);
    }

    graphInfoDst->inputTensors = nullptr;
    graphInfoDst->numInputTensors = 0;
    if (graphInfoSrc->graphInputs) {
        if (!copyTensorsInfo(graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
            return false;
        }
        graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
    }

    graphInfoDst->outputTensors = nullptr;
    graphInfoDst->numOutputTensors = 0;
    if (graphInfoSrc->graphOutputs) {
        if (!copyTensorsInfo(graphInfoSrc->graphOutputs, graphInfoDst->outputTensors, graphInfoSrc->numGraphOutputs)) {
            return false;
        }
        graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
    }

    return true;
}

// Helper to copy graph info V3
static bool copyGraphsInfoV3(const QnnSystemContext_GraphInfoV3_t* graphInfoSrc, GraphInfo_t* graphInfoDst) {
    graphInfoDst->graphName = nullptr;
    if (graphInfoSrc->graphName) {
        graphInfoDst->graphName = strdup(graphInfoSrc->graphName);
    }

    graphInfoDst->inputTensors = nullptr;
    graphInfoDst->numInputTensors = 0;
    if (graphInfoSrc->graphInputs) {
        if (!copyTensorsInfo(graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
            return false;
        }
        graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
    }

    graphInfoDst->outputTensors = nullptr;
    graphInfoDst->numOutputTensors = 0;
    if (graphInfoSrc->graphOutputs) {
        if (!copyTensorsInfo(graphInfoSrc->graphOutputs, graphInfoDst->outputTensors, graphInfoSrc->numGraphOutputs)) {
            return false;
        }
        graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
    }

    return true;
}

// Helper to free tensor resources
static void freeQnnTensor(Qnn_Tensor_t& tensor) {
    if (QNN_TENSOR_GET_NAME(tensor)) {
        free((void*)QNN_TENSOR_GET_NAME(tensor));
    }
    if (QNN_TENSOR_GET_DIMENSIONS(tensor)) {
        free(QNN_TENSOR_GET_DIMENSIONS(tensor));
    }
}

class VEGCombinedRunner {
public:
    VEGCombinedRunner(const std::string& backendPath,
                      const std::string& systemLibPath,
                      const std::string& serializedBinPath,
                      const std::string& imagePath,
                      const std::string& outputDir,
                      int32_t targetSize,
                      bool useOpenCL,
                      const std::string& framePath = "",
                      int32_t frameWidth = 0,
                      int32_t frameHeight = 0,
                      int32_t frameStride = 0)
        : m_backendPath(backendPath),
          m_systemLibPath(systemLibPath),
          m_serializedBinPath(serializedBinPath),
          m_imagePath(imagePath),
          m_outputDir(outputDir),
          m_targetSize(targetSize),
          m_useOpenCL(useOpenCL),
          m_framePath(framePath),
          m_frameWidth(frameWidth),
          m_frameHeight(frameHeight),
          m_frameStride(frameStride),
          m_backendLibHandle(nullptr),
          m_systemLibHandle(nullptr),
          m_backendHandle(nullptr),
          m_deviceHandle(nullptr),
          m_context(nullptr),
          m_graphInfo(nullptr),
          m_graphsCount(0) {
        memset(&m_qnnInterface, 0, sizeof(m_qnnInterface));
        memset(&m_qnnSystemInterface, 0, sizeof(m_qnnSystemInterface));
    }

    ~VEGCombinedRunner() {
        cleanup();
    }

    bool initialize() {
        LOG_INFO("Initializing VEG Combined Runner...");

        // Create output directory
        if (!createDirectory(m_outputDir)) {
            LOG_ERROR("Failed to create output directory: %s", m_outputDir.c_str());
            return false;
        }

        // Load backend library
        if (!loadBackendLibrary()) {
            LOG_ERROR("Failed to load backend library");
            return false;
        }

        // Load system library
        if (!loadSystemLibrary()) {
            LOG_ERROR("Failed to load system library");
            return false;
        }

        // Initialize backend
        if (!initializeBackend()) {
            LOG_ERROR("Failed to initialize backend");
            return false;
        }

        // Create device
        if (!createDevice()) {
            LOG_ERROR("Failed to create device");
            return false;
        }

        // Load context from binary
        if (!loadContextFromBinary()) {
            LOG_ERROR("Failed to load context from binary");
            return false;
        }

        LOG_INFO("Initialization complete!");
        return true;
    }

    bool execute() {
        LOG_INFO("=== Stage 1: Image Preprocessing (OpenCL) ===");

        auto preprocessStart = std::chrono::high_resolution_clock::now();

        // Initialize preprocessor with Qwen3-VL config
        qwen_vl::QwenVLConfig config;
        config.patch_size = 16;            // Qwen3-VL: 16x16 patches
        config.temporal_patch_size = 2;
        config.merge_size = 2;
        config.min_image_tokens = 4;
        config.max_image_tokens = 16384;
        config.vit_pos_emb_dim = 32;       // Qwen3-VL: h+w only (32 dims)
        config.image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
        config.image_std = {0.26862954f, 0.26130258f, 0.27577711f};

        qwen_vl::QwenVLPreprocessor preprocessor(config, m_useOpenCL);

        // Preprocess: frame mode (DMA-BUF RGB24) or image mode (file)
        qwen_vl::PreprocessedData preprocessed;
        if (!m_framePath.empty()) {
            // Load raw RGB24 frame data from file
            std::ifstream frameFile(m_framePath, std::ios::binary | std::ios::ate);
            if (!frameFile.is_open()) {
                LOG_ERROR("Failed to open frame file: %s", m_framePath.c_str());
                return false;
            }
            size_t fileSize = frameFile.tellg();
            frameFile.seekg(0);
            int32_t stride = m_frameStride > 0 ? m_frameStride : m_frameWidth * 3;
            size_t expectedSize = (size_t)m_frameHeight * stride;
            if (fileSize < expectedSize) {
                LOG_ERROR("Frame file too small: %zu < %zu", fileSize, expectedSize);
                return false;
            }
            std::vector<uint8_t> frameData(expectedSize);
            frameFile.read(reinterpret_cast<char*>(frameData.data()), expectedSize);
            frameFile.close();

            LOG_INFO("Frame mode: %dx%d stride=%d (%zu bytes)", m_frameWidth, m_frameHeight, stride, expectedSize);
            preprocessed = preprocessor.preprocessFromFrame(
                frameData.data(), m_frameWidth, m_frameHeight, stride, m_targetSize);
        } else {
            preprocessed = preprocessor.preprocessImage(m_imagePath, m_targetSize);
        }

        auto preprocessEnd = std::chrono::high_resolution_clock::now();
        auto preprocessDuration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocessEnd - preprocessStart);

        LOG_INFO("Preprocessing complete in %ld ms", preprocessDuration.count());
        LOG_INFO("  Sequence length: %d", preprocessed.seq_len);
        LOG_INFO("  Resized dimensions: %dx%d", preprocessed.resized_height, preprocessed.resized_width);
        LOG_INFO("  Grid THW: [%d, %d, %d]", preprocessed.image_grid_thw[0],
                 preprocessed.image_grid_thw[1], preprocessed.image_grid_thw[2]);

        LOG_INFO("\n=== Stage 2: Vision Encoder Inference (QNN) ===");

        auto inferenceStart = std::chrono::high_resolution_clock::now();

        if (m_graphsCount == 0) {
            LOG_ERROR("No graphs loaded");
            return false;
        }

        uint32_t graphIdx = 0;
        GraphInfo_t& graphInfo = m_graphInfo[graphIdx];

        // Prepare input tensors from preprocessed data (directly in memory)
        Qnn_Tensor_t* inputs = nullptr;
        if (!prepareInputTensorsFromPreprocessed(&inputs, graphIdx, preprocessed)) {
            LOG_ERROR("Failed to prepare input tensors");
            return false;
        }

        // Prepare output tensors
        Qnn_Tensor_t* outputs = nullptr;
        if (!prepareOutputTensors(&outputs, graphIdx)) {
            LOG_ERROR("Failed to prepare output tensors");
            tearDownTensors(inputs, graphInfo.numInputTensors);
            return false;
        }

        LOG_INFO("Executing graph: %s", graphInfo.graphName ? graphInfo.graphName : "unnamed");

        // Execute graph
        Qnn_ErrorHandle_t error = m_qnnInterface.QNN_INTERFACE_VER_NAME.graphExecute(
            graphInfo.graph,
            inputs,
            graphInfo.numInputTensors,
            outputs,
            graphInfo.numOutputTensors,
            nullptr,
            nullptr);

        if (error != QNN_SUCCESS) {
            LOG_ERROR("Graph execution failed with error: %lu", (unsigned long)error);
            tearDownTensors(inputs, graphInfo.numInputTensors);
            tearDownTensors(outputs, graphInfo.numOutputTensors);
            return false;
        }

        auto inferenceEnd = std::chrono::high_resolution_clock::now();
        auto inferenceDuration = std::chrono::duration_cast<std::chrono::milliseconds>(inferenceEnd - inferenceStart);

        LOG_INFO("Inference complete in %ld ms", inferenceDuration.count());

        // Save outputs
        if (!saveOutputs(outputs, graphInfo.numOutputTensors)) {
            LOG_ERROR("Failed to save outputs");
            tearDownTensors(inputs, graphInfo.numInputTensors);
            tearDownTensors(outputs, graphInfo.numOutputTensors);
            return false;
        }

        // Cleanup tensors
        tearDownTensors(inputs, graphInfo.numInputTensors);
        tearDownTensors(outputs, graphInfo.numOutputTensors);

        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(inferenceEnd - preprocessStart);
        LOG_INFO("\n=== Total Execution Time: %ld ms ===", totalDuration.count());

        return true;
    }

private:
    // Configuration
    std::string m_backendPath;
    std::string m_systemLibPath;
    std::string m_serializedBinPath;
    std::string m_imagePath;
    std::string m_outputDir;
    int32_t m_targetSize;
    bool m_useOpenCL;

    // Frame mode (DMA-BUF RGB24 input)
    std::string m_framePath;
    int32_t m_frameWidth;
    int32_t m_frameHeight;
    int32_t m_frameStride;

    // Library handles
    void* m_backendLibHandle;
    void* m_systemLibHandle;

    // QNN handles
    Qnn_BackendHandle_t m_backendHandle;
    Qnn_DeviceHandle_t m_deviceHandle;
    Qnn_ContextHandle_t m_context;

    // Graph info
    GraphInfo_t* m_graphInfo;
    uint32_t m_graphsCount;

    // QNN interfaces
    QnnInterface_t m_qnnInterface;
    QnnSystemInterface_t m_qnnSystemInterface;

    bool createDirectory(const std::string& path) {
        struct stat st;
        if (stat(path.c_str(), &st) == 0) {
            return S_ISDIR(st.st_mode);
        }
        return mkdir(path.c_str(), 0755) == 0;
    }

    bool loadBackendLibrary() {
        LOG_INFO("Loading backend library: %s", m_backendPath.c_str());

        m_backendLibHandle = dlopen(m_backendPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!m_backendLibHandle) {
            LOG_ERROR("Failed to load backend: %s", dlerror());
            return false;
        }

        QnnInterfaceGetProvidersFn_t getProviders =
            (QnnInterfaceGetProvidersFn_t)dlsym(m_backendLibHandle, "QnnInterface_getProviders");

        if (!getProviders) {
            LOG_ERROR("Failed to get QnnInterface_getProviders: %s", dlerror());
            return false;
        }

        const QnnInterface_t** providerList = nullptr;
        uint32_t numProviders = 0;

        if (getProviders(&providerList, &numProviders) != QNN_SUCCESS || numProviders == 0) {
            LOG_ERROR("Failed to get providers");
            return false;
        }

        m_qnnInterface = *providerList[0];
        LOG_INFO("Backend library loaded successfully");
        return true;
    }

    bool loadSystemLibrary() {
        LOG_INFO("Loading system library: %s", m_systemLibPath.c_str());

        m_systemLibHandle = dlopen(m_systemLibPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!m_systemLibHandle) {
            LOG_ERROR("Failed to load system library: %s", dlerror());
            return false;
        }

        QnnSystemInterfaceGetProvidersFn_t getProviders =
            (QnnSystemInterfaceGetProvidersFn_t)dlsym(m_systemLibHandle,
                                                      "QnnSystemInterface_getProviders");

        if (!getProviders) {
            LOG_ERROR("Failed to get QnnSystemInterface_getProviders: %s", dlerror());
            return false;
        }

        const QnnSystemInterface_t** providerList = nullptr;
        uint32_t numProviders = 0;

        if (getProviders(&providerList, &numProviders) != QNN_SUCCESS || numProviders == 0) {
            LOG_ERROR("Failed to get system providers");
            return false;
        }

        m_qnnSystemInterface = *providerList[0];
        LOG_INFO("System library loaded successfully");
        return true;
    }

    bool initializeBackend() {
        LOG_INFO("Initializing backend...");

        Qnn_ErrorHandle_t error = m_qnnInterface.QNN_INTERFACE_VER_NAME.backendCreate(nullptr, nullptr, &m_backendHandle);
        if (error != QNN_SUCCESS) {
            LOG_ERROR("Failed to create backend: %lu", (unsigned long)error);
            return false;
        }

        LOG_INFO("Backend initialized successfully");
        return true;
    }

    bool createDevice() {
        LOG_INFO("Creating device...");

        if (!m_qnnInterface.QNN_INTERFACE_VER_NAME.deviceCreate) {
            LOG_WARN("Device creation not supported by backend");
            return true;
        }

        // QCS9075 (SA8775P class): soc_id=52, Hexagon arch v73 — must match Stage 2 context binary
        QnnHtpDevice_CustomConfig_t htpSocConfig;
        htpSocConfig.option   = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
        htpSocConfig.socModel = 52;

        QnnHtpDevice_CustomConfig_t htpArchConfig;
        htpArchConfig.option        = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
        htpArchConfig.arch.deviceId = 0;
        htpArchConfig.arch.arch     = QNN_HTP_DEVICE_ARCH_V73;

        QnnHtpDevice_CustomConfig_t htpPdConfig;
        htpPdConfig.option                                  = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
        htpPdConfig.useSignedProcessDomain.deviceId         = 0;
        htpPdConfig.useSignedProcessDomain.useSignedProcessDomain = false;  // unsigned PD

        QnnDevice_Config_t devCfg0 = {QNN_DEVICE_CONFIG_OPTION_CUSTOM, {&htpSocConfig}};
        QnnDevice_Config_t devCfg1 = {QNN_DEVICE_CONFIG_OPTION_CUSTOM, {&htpArchConfig}};
        QnnDevice_Config_t devCfg2 = {QNN_DEVICE_CONFIG_OPTION_CUSTOM, {&htpPdConfig}};
        const QnnDevice_Config_t* devConfigs[] = {&devCfg0, &devCfg1, &devCfg2, nullptr};

        Qnn_ErrorHandle_t error = m_qnnInterface.QNN_INTERFACE_VER_NAME.deviceCreate(
            nullptr, devConfigs, &m_deviceHandle);
        if (error != QNN_SUCCESS) {
            LOG_WARN("deviceCreate returned %lu — proceeding with null device handle", (unsigned long)error);
            m_deviceHandle = nullptr;
        } else {
            LOG_INFO("Device created successfully");
        }
        return true;
    }

    bool loadContextFromBinary() {
        LOG_INFO("Loading context from binary: %s", m_serializedBinPath.c_str());

        std::ifstream file(m_serializedBinPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open binary file: %s", m_serializedBinPath.c_str());
            return false;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            LOG_ERROR("Failed to read binary file");
            return false;
        }
        file.close();

        LOG_INFO("Binary file size: %ld bytes", size);

        if (!m_qnnSystemInterface.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate ||
            !m_qnnSystemInterface.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetBinaryInfo ||
            !m_qnnSystemInterface.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree) {
            LOG_ERROR("QNN System function pointers are not populated");
            return false;
        }

        QnnSystemContext_Handle_t sysCtxHandle = nullptr;
        Qnn_ErrorHandle_t error = m_qnnSystemInterface.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate(&sysCtxHandle);
        if (error != QNN_SUCCESS) {
            LOG_ERROR("Could not create system context handle: %lu", (unsigned long)error);
            return false;
        }

        const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
        Qnn_ContextBinarySize_t binaryInfoSize = 0;
        error = m_qnnSystemInterface.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetBinaryInfo(
            sysCtxHandle, buffer.data(), size, &binaryInfo, &binaryInfoSize);
        if (error != QNN_SUCCESS) {
            LOG_ERROR("Failed to get context binary info: %lu", (unsigned long)error);
            m_qnnSystemInterface.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(sysCtxHandle);
            return false;
        }

        if (!extractGraphsInfo(binaryInfo)) {
            LOG_ERROR("Failed to extract graphs info from binary");
            m_qnnSystemInterface.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(sysCtxHandle);
            return false;
        }

        m_qnnSystemInterface.QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(sysCtxHandle);

        LOG_INFO("Extracted %u graphs from binary", m_graphsCount);
        for (uint32_t i = 0; i < m_graphsCount; i++) {
            LOG_INFO("  Graph %u: %s (%u inputs, %u outputs)",
                     i, m_graphInfo[i].graphName ? m_graphInfo[i].graphName : "unnamed",
                     m_graphInfo[i].numInputTensors, m_graphInfo[i].numOutputTensors);
        }

        // v73: no extended UDMA. Optional graph-level options go in const QnnContext_Config_t** (nullptr = default).
        const QnnContext_Config_t** ctxConfigs = nullptr;

        error = m_qnnInterface.QNN_INTERFACE_VER_NAME.contextCreateFromBinary(
            m_backendHandle,
            m_deviceHandle,
            ctxConfigs,
            buffer.data(),
            size,
            &m_context,
            nullptr);

        if (error != QNN_SUCCESS) {
            LOG_ERROR("Failed to create context from binary: %lu", (unsigned long)error);
            return false;
        }

        for (uint32_t i = 0; i < m_graphsCount; i++) {
            error = m_qnnInterface.QNN_INTERFACE_VER_NAME.graphRetrieve(
                m_context, m_graphInfo[i].graphName, &m_graphInfo[i].graph);
            if (error != QNN_SUCCESS) {
                LOG_ERROR("Failed to retrieve graph handle for '%s': %lu",
                          m_graphInfo[i].graphName ? m_graphInfo[i].graphName : "unnamed",
                          (unsigned long)error);
                return false;
            }
            LOG_INFO("Retrieved graph handle for: %s", m_graphInfo[i].graphName);
        }

        LOG_INFO("Context loaded successfully from binary");
        return true;
    }

    bool extractGraphsInfo(const QnnSystemContext_BinaryInfo_t* binaryInfo) {
        if (!binaryInfo) {
            LOG_ERROR("binaryInfo is nullptr");
            return false;
        }

        const QnnSystemContext_GraphInfo_t* graphs = nullptr;
        uint32_t numGraphs = 0;

        if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
            graphs = binaryInfo->contextBinaryInfoV1.graphs;
            numGraphs = binaryInfo->contextBinaryInfoV1.numGraphs;
        } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
            graphs = binaryInfo->contextBinaryInfoV2.graphs;
            numGraphs = binaryInfo->contextBinaryInfoV2.numGraphs;
        } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
            graphs = binaryInfo->contextBinaryInfoV3.graphs;
            numGraphs = binaryInfo->contextBinaryInfoV3.numGraphs;
        } else {
            LOG_ERROR("Unrecognized binary info version: %d", binaryInfo->version);
            return false;
        }

        if (!graphs || numGraphs == 0) {
            LOG_ERROR("No graphs found in binary");
            return false;
        }

        m_graphInfo = (GraphInfo_t*)calloc(numGraphs, sizeof(GraphInfo_t));
        if (!m_graphInfo) {
            LOG_ERROR("Failed to allocate memory for graph info");
            return false;
        }
        m_graphsCount = numGraphs;

        for (uint32_t i = 0; i < numGraphs; i++) {
            if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
                if (!copyGraphsInfoV1(&graphs[i].graphInfoV1, &m_graphInfo[i])) {
                    LOG_ERROR("Failed to copy graph info for graph %u", i);
                    return false;
                }
            } else if (graphs[i].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
                if (!copyGraphsInfoV3(&graphs[i].graphInfoV3, &m_graphInfo[i])) {
                    LOG_ERROR("Failed to copy graph info V3 for graph %u", i);
                    return false;
                }
            } else {
                LOG_ERROR("Unsupported graph info version: %d", graphs[i].version);
                return false;
            }
        }

        return true;
    }

    // NEW: Prepare input tensors directly from preprocessed data (no file I/O)
    bool prepareInputTensorsFromPreprocessed(Qnn_Tensor_t** inputs, uint32_t graphIdx,
                                              const qwen_vl::PreprocessedData& preprocessed) {
        LOG_INFO("Preparing input tensors from preprocessed data...");

        if (graphIdx >= m_graphsCount) {
            LOG_ERROR("Invalid graph index: %u", graphIdx);
            return false;
        }

        GraphInfo_t& graphInfo = m_graphInfo[graphIdx];
        uint32_t numInputs = graphInfo.numInputTensors;

        *inputs = (Qnn_Tensor_t*)calloc(numInputs, sizeof(Qnn_Tensor_t));
        if (!*inputs) {
            LOG_ERROR("Failed to allocate input tensors");
            return false;
        }

        // Map tensor names to preprocessed data. Stage 1 ONNX exports:
        // hidden_states, position_embeddings_cos, position_embeddings_sin (see qwen3_vl_vision.py).
        std::map<std::string, const std::vector<float>*> dataMap;
        dataMap["hidden_states"] = &preprocessed.pixel_values;
        dataMap["position_embeddings_cos"] = &preprocessed.position_ids_cos;
        dataMap["position_embeddings_sin"] = &preprocessed.position_ids_sin;
        // Legacy/alternate naming
        dataMap["pixel_values"] = &preprocessed.pixel_values;
        dataMap["position_ids_cos"] = &preprocessed.position_ids_cos;
        dataMap["position_ids_sin"] = &preprocessed.position_ids_sin;

        for (uint32_t i = 0; i < numInputs; i++) {
            Qnn_Tensor_t& srcTensor = graphInfo.inputTensors[i];
            Qnn_Tensor_t& dstTensor = (*inputs)[i];

            dstTensor = QNN_TENSOR_INIT;
            deepCopyQnnTensorInfo(&dstTensor, &srcTensor);
            QNN_TENSOR_SET_MEM_TYPE(&dstTensor, QNN_TENSORMEMTYPE_RAW);

            size_t tensorSize = calculateTensorSize(&srcTensor);
            if (tensorSize == 0) {
                LOG_ERROR("Failed to calculate tensor size for input %u", i);
                return false;
            }

            const char* tensorName = QNN_TENSOR_GET_NAME(&srcTensor);
            if (!tensorName) {
                LOG_ERROR("Tensor name is null for input %u", i);
                return false;
            }

            auto it = dataMap.find(tensorName);
            if (it == dataMap.end()) {
                LOG_ERROR("No preprocessed data found for tensor: %s", tensorName);
                return false;
            }

            const std::vector<float>* srcData = it->second;
            Qnn_DataType_t dataType = QNN_TENSOR_GET_DATA_TYPE(&srcTensor);

            // Allocate buffer
            uint8_t* data = (uint8_t*)malloc(tensorSize);
            if (!data) {
                LOG_ERROR("Failed to allocate buffer for input tensor");
                return false;
            }

            // Convert data based on tensor data type
            if (dataType == QNN_DATATYPE_UFIXED_POINT_16 || dataType == QNN_DATATYPE_SFIXED_POINT_16) {
                // Quantize FP32 to uint16
                Qnn_QuantizeParams_t qParams = QNN_TENSOR_GET_QUANT_PARAMS(&srcTensor);
                float scale = 1.0f;
                int32_t offset = 0;

                if (qParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                    scale = qParams.scaleOffsetEncoding.scale;
                    offset = qParams.scaleOffsetEncoding.offset;
                }

                uint16_t* quantizedData = reinterpret_cast<uint16_t*>(data);
                for (size_t j = 0; j < srcData->size(); j++) {
                    quantizedData[j] = quantizeToUint16((*srcData)[j], scale, offset);
                }
                LOG_INFO("  Quantized %s: %zu elements (scale=%f, offset=%d)",
                         tensorName, srcData->size(), scale, offset);
            } else if (dataType == QNN_DATATYPE_FLOAT_16) {
                // Convert FP32 to FP16
                uint16_t* fp16Data = reinterpret_cast<uint16_t*>(data);
                for (size_t j = 0; j < srcData->size(); j++) {
                    fp16Data[j] = float32ToFloat16((*srcData)[j]);
                }
                LOG_INFO("  Converted %s: %zu elements to FP16", tensorName, srcData->size());
            } else if (dataType == QNN_DATATYPE_FLOAT_32) {
                // Direct copy
                memcpy(data, srcData->data(), srcData->size() * sizeof(float));
                LOG_INFO("  Copied %s: %zu elements as FP32", tensorName, srcData->size());
            } else {
                LOG_ERROR("Unsupported data type %d for tensor %s", dataType, tensorName);
                free(data);
                return false;
            }

            Qnn_ClientBuffer_t clientBuf;
            clientBuf.data = data;
            clientBuf.dataSize = tensorSize;
            QNN_TENSOR_SET_CLIENT_BUF(&dstTensor, clientBuf);
        }

        return true;
    }

    bool prepareOutputTensors(Qnn_Tensor_t** outputs, uint32_t graphIdx) {
        LOG_INFO("Preparing output tensors for graph %u...", graphIdx);

        if (graphIdx >= m_graphsCount) {
            LOG_ERROR("Invalid graph index: %u", graphIdx);
            return false;
        }

        GraphInfo_t& graphInfo = m_graphInfo[graphIdx];
        uint32_t numOutputs = graphInfo.numOutputTensors;

        *outputs = (Qnn_Tensor_t*)calloc(numOutputs, sizeof(Qnn_Tensor_t));
        if (!*outputs) {
            LOG_ERROR("Failed to allocate output tensors");
            return false;
        }

        for (uint32_t i = 0; i < numOutputs; i++) {
            Qnn_Tensor_t& srcTensor = graphInfo.outputTensors[i];
            Qnn_Tensor_t& dstTensor = (*outputs)[i];

            dstTensor = QNN_TENSOR_INIT;
            deepCopyQnnTensorInfo(&dstTensor, &srcTensor);
            QNN_TENSOR_SET_MEM_TYPE(&dstTensor, QNN_TENSORMEMTYPE_RAW);

            size_t tensorSize = calculateTensorSize(&srcTensor);
            if (tensorSize == 0) {
                LOG_ERROR("Failed to calculate tensor size for output %u", i);
                return false;
            }

            uint8_t* data = (uint8_t*)malloc(tensorSize);
            if (!data) {
                LOG_ERROR("Failed to allocate buffer for output tensor");
                return false;
            }
            memset(data, 0, tensorSize);

            Qnn_ClientBuffer_t clientBuf;
            clientBuf.data = data;
            clientBuf.dataSize = tensorSize;
            QNN_TENSOR_SET_CLIENT_BUF(&dstTensor, clientBuf);

            const char* tensorName = QNN_TENSOR_GET_NAME(&srcTensor);
            LOG_INFO("  Prepared output tensor %u: %s (size=%zu bytes)",
                     i, tensorName ? tensorName : "unnamed", tensorSize);
        }

        return true;
    }

    void tearDownTensors(Qnn_Tensor_t* tensors, uint32_t count) {
        if (!tensors) return;

        for (uint32_t i = 0; i < count; i++) {
            if (QNN_TENSOR_GET_NAME(tensors[i])) {
                free((void*)QNN_TENSOR_GET_NAME(tensors[i]));
            }
            if (QNN_TENSOR_GET_DIMENSIONS(tensors[i])) {
                free(QNN_TENSOR_GET_DIMENSIONS(tensors[i]));
            }
            if (QNN_TENSOR_GET_CLIENT_BUF(tensors[i]).data) {
                free(QNN_TENSOR_GET_CLIENT_BUF(tensors[i]).data);
            }
        }
        free(tensors);
    }

    bool saveOutputs(Qnn_Tensor_t* outputs, uint32_t numOutputs) {
        LOG_INFO("Saving outputs to %s", m_outputDir.c_str());

        for (uint32_t i = 0; i < numOutputs; i++) {
            const Qnn_Tensor_t& tensor = outputs[i];
            const char* tensorName = QNN_TENSOR_GET_NAME(&tensor);

            std::string filename;
            if (tensorName && strlen(tensorName) > 0) {
                filename = m_outputDir + "/" + std::string(tensorName) + ".raw";
            } else {
                filename = m_outputDir + "/output_" + std::to_string(i) + ".raw";
            }

            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                LOG_ERROR("Failed to open output file: %s", filename.c_str());
                return false;
            }

            Qnn_ClientBuffer_t clientBuf = QNN_TENSOR_GET_CLIENT_BUF(&tensor);
            Qnn_DataType_t dataType = QNN_TENSOR_GET_DATA_TYPE(&tensor);

            // Dequantize if needed
            if (dataType == QNN_DATATYPE_UFIXED_POINT_16 || dataType == QNN_DATATYPE_SFIXED_POINT_16) {
                size_t numElements = clientBuf.dataSize / sizeof(uint16_t);
                std::vector<float> fp32Data(numElements);
                const uint16_t* quantizedData = reinterpret_cast<const uint16_t*>(clientBuf.data);

                Qnn_QuantizeParams_t qParams = QNN_TENSOR_GET_QUANT_PARAMS(&tensor);
                float scale = 1.0f;
                int32_t offset = 0;

                if (qParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                    scale = qParams.scaleOffsetEncoding.scale;
                    offset = qParams.scaleOffsetEncoding.offset;
                }

                for (size_t j = 0; j < numElements; j++) {
                    fp32Data[j] = dequantizeFromUint16(quantizedData[j], scale, offset);
                }

                file.write(reinterpret_cast<const char*>(fp32Data.data()), numElements * sizeof(float));
                LOG_INFO("  Saved: %s (%zu bytes, dequantized)", filename.c_str(), numElements * sizeof(float));
            } else if (dataType == QNN_DATATYPE_FLOAT_16) {
                size_t numElements = clientBuf.dataSize / sizeof(uint16_t);
                std::vector<float> fp32Data(numElements);
                const uint16_t* fp16Data = reinterpret_cast<const uint16_t*>(clientBuf.data);

                for (size_t j = 0; j < numElements; j++) {
                    fp32Data[j] = float16ToFloat32(fp16Data[j]);
                }

                file.write(reinterpret_cast<const char*>(fp32Data.data()), numElements * sizeof(float));
                LOG_INFO("  Saved: %s (%zu bytes, FP16->FP32)", filename.c_str(), numElements * sizeof(float));
            } else {
                file.write(reinterpret_cast<const char*>(clientBuf.data), clientBuf.dataSize);
                LOG_INFO("  Saved: %s (%u bytes)", filename.c_str(), clientBuf.dataSize);
            }
            file.close();
        }

        return true;
    }

    void cleanup() {
        LOG_INFO("Cleaning up...");

        if (m_graphInfo) {
            for (uint32_t i = 0; i < m_graphsCount; i++) {
                if (m_graphInfo[i].graphName) {
                    free(m_graphInfo[i].graphName);
                }
                if (m_graphInfo[i].inputTensors) {
                    for (uint32_t j = 0; j < m_graphInfo[i].numInputTensors; j++) {
                        freeQnnTensor(m_graphInfo[i].inputTensors[j]);
                    }
                    free(m_graphInfo[i].inputTensors);
                }
                if (m_graphInfo[i].outputTensors) {
                    for (uint32_t j = 0; j < m_graphInfo[i].numOutputTensors; j++) {
                        freeQnnTensor(m_graphInfo[i].outputTensors[j]);
                    }
                    free(m_graphInfo[i].outputTensors);
                }
            }
            free(m_graphInfo);
            m_graphInfo = nullptr;
        }

        if (m_context && m_qnnInterface.QNN_INTERFACE_VER_NAME.contextFree) {
            m_qnnInterface.QNN_INTERFACE_VER_NAME.contextFree(m_context, nullptr);
        }

        if (m_deviceHandle && m_qnnInterface.QNN_INTERFACE_VER_NAME.deviceFree) {
            m_qnnInterface.QNN_INTERFACE_VER_NAME.deviceFree(m_deviceHandle);
        }

        if (m_backendHandle && m_qnnInterface.QNN_INTERFACE_VER_NAME.backendFree) {
            m_qnnInterface.QNN_INTERFACE_VER_NAME.backendFree(m_backendHandle);
        }

        if (m_backendLibHandle) {
            dlclose(m_backendLibHandle);
        }

        if (m_systemLibHandle) {
            dlclose(m_systemLibHandle);
        }
    }
};

void showHelp() {
    std::cout << "\nVEG Combined Runner - Image to VEG Encoding in one step\n";
    std::cout << "\nUSAGE:\n";
    std::cout << "  veg_combined_runner [OPTIONS]\n\n";
    std::cout << "OPTIONS:\n";
    std::cout << "  --backend <FILE>          Path to QNN backend library (default: /usr/lib/libQnnHtp.so)\n";
    std::cout << "  --system_library <FILE>   Path to QNN system library (default: /usr/lib/libQnnSystem.so)\n";
    std::cout << "  --model <FILE>            Path to serialized binary (veg.serialized.bin)\n";
    std::cout << "  --image <FILE>            Path to input image (jpg/png)\n";
    std::cout << "\nFRAME MODE (DMA-BUF RGB24 input, replaces --image):\n";
    std::cout << "  --frame <FILE>            Path to raw RGB24 frame data\n";
    std::cout << "  --frame_width <INT>       Frame width in pixels (required with --frame)\n";
    std::cout << "  --frame_height <INT>      Frame height in pixels (required with --frame)\n";
    std::cout << "  --frame_stride <INT>      Row stride in bytes (default: width*3)\n";
    std::cout << "\nOPTIONAL OPTIONS:\n";
    std::cout << "  --output_dir <DIR>        Output directory (default: ./output)\n";
    std::cout << "  --target_size <SIZE>      Target image size (default: 448)\n";
    std::cout << "  --no_opencl               Disable OpenCL acceleration\n";
    std::cout << "  --help                    Show this help message\n";
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    std::string backendPath = "/usr/lib/libQnnHtp.so";
    std::string systemLibPath = "/usr/lib/libQnnSystem.so";
    std::string serializedBinPath;
    std::string imagePath;
    std::string outputDir = "./output";
    int32_t targetSize = 448;
    bool useOpenCL = true;
    std::string framePath;
    int32_t frameWidth = 0;
    int32_t frameHeight = 0;
    int32_t frameStride = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            showHelp();
            return 0;
        } else if (arg == "--backend" && i + 1 < argc) {
            backendPath = argv[++i];
        } else if (arg == "--system_library" && i + 1 < argc) {
            systemLibPath = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            serializedBinPath = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            imagePath = argv[++i];
        } else if (arg == "--frame" && i + 1 < argc) {
            framePath = argv[++i];
        } else if (arg == "--frame_width" && i + 1 < argc) {
            frameWidth = std::stoi(argv[++i]);
        } else if (arg == "--frame_height" && i + 1 < argc) {
            frameHeight = std::stoi(argv[++i]);
        } else if (arg == "--frame_stride" && i + 1 < argc) {
            frameStride = std::stoi(argv[++i]);
        } else if (arg == "--output_dir" && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--target_size" && i + 1 < argc) {
            targetSize = std::stoi(argv[++i]);
        } else if (arg == "--no_opencl") {
            useOpenCL = false;
        } else {
            LOG_ERROR("Unknown argument: %s", arg.c_str());
            showHelp();
            return 1;
        }
    }

    // Validate required arguments
    if (backendPath.empty() || systemLibPath.empty() || serializedBinPath.empty()) {
        LOG_ERROR("Missing required arguments: --backend, --system_library, --model");
        showHelp();
        return 1;
    }
    if (framePath.empty() && imagePath.empty()) {
        LOG_ERROR("Must specify either --image or --frame");
        showHelp();
        return 1;
    }
    if (!framePath.empty() && (frameWidth <= 0 || frameHeight <= 0)) {
        LOG_ERROR("--frame requires --frame_width and --frame_height");
        showHelp();
        return 1;
    }

    LOG_INFO("========================================");
    LOG_INFO("  VEG Combined Runner");
    LOG_INFO("========================================");
    LOG_INFO("Backend: %s", backendPath.c_str());
    LOG_INFO("System Library: %s", systemLibPath.c_str());
    LOG_INFO("Model: %s", serializedBinPath.c_str());
    if (!framePath.empty()) {
        LOG_INFO("Frame: %s (%dx%d stride=%d)", framePath.c_str(), frameWidth, frameHeight, frameStride);
    } else {
        LOG_INFO("Image: %s", imagePath.c_str());
    }
    LOG_INFO("Output Directory: %s", outputDir.c_str());
    LOG_INFO("Target Size: %d", targetSize);
    LOG_INFO("OpenCL: %s", useOpenCL ? "Enabled" : "Disabled");
    LOG_INFO("========================================\n");

    VEGCombinedRunner runner(backendPath, systemLibPath, serializedBinPath,
                             imagePath, outputDir, targetSize, useOpenCL,
                             framePath, frameWidth, frameHeight, frameStride);

    if (!runner.initialize()) {
        LOG_ERROR("Failed to initialize");
        return 1;
    }

    if (!runner.execute()) {
        LOG_ERROR("Failed to execute");
        return 1;
    }

    LOG_INFO("\nVEG encoding completed successfully!");
    LOG_INFO("Output saved to: %s", outputDir.c_str());
    return 0;
}
