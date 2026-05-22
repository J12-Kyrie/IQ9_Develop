//==============================================================================
//
//  VEG Server Combined (New Rope)
//  Unified server that combines VEGCombinedRunner with Unix socket server
//  Keeps the model loaded and reuses it for multiple inference requests
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
#include <sys/socket.h>
#include <sys/un.h>
#include <signal.h>
#include <cmath>
#include <chrono>
#include <errno.h>

// QNN Headers
#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/QnnCommon.h"
#include "QNN/QnnDevice.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/System/QnnSystemInterface.h"
#include "QNN/System/QnnSystemContext.h"

// OpenCL Preprocessor
#include "qwen_vl_preprocessor.h"

#define SOCKET_PATH "/data/local/tmp/veg_server.sock"
#define LOG_INFO(fmt, ...) fprintf(stdout, "[VEG-SERVER] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[VEG-SERVER ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) fprintf(stdout, "[VEG-SERVER WARN] " fmt "\n", ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) fprintf(stdout, "[VEG-SERVER DEBUG] " fmt "\n", ##__VA_ARGS__)

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

//==============================================================================
// VEGServerCombined - Unified server with persistent model loading
//==============================================================================
class VEGServerCombined {
public:
    VEGServerCombined(const std::string& backendPath,
                      const std::string& systemLibPath,
                      const std::string& modelPath,
                      int32_t targetSize = 448,
                      bool useOpenCL = true)
        : m_backendPath(backendPath),
          m_systemLibPath(systemLibPath),
          m_modelPath(modelPath),
          m_targetSize(targetSize),
          m_useOpenCL(useOpenCL),
          m_serverSocket(-1),
          m_running(false),
          m_backendLibHandle(nullptr),
          m_systemLibHandle(nullptr),
          m_backendHandle(nullptr),
          m_deviceHandle(nullptr),
          m_context(nullptr),
          m_graphInfo(nullptr),
          m_graphsCount(0),
          m_preprocessor(nullptr) {
        memset(&m_qnnInterface, 0, sizeof(m_qnnInterface));
        memset(&m_qnnSystemInterface, 0, sizeof(m_qnnSystemInterface));
    }

    ~VEGServerCombined() {
        stop();
        cleanupQNN();
    }

    bool initialize() {
        LOG_INFO("Initializing VEG Server Combined (New Rope)...");

        // Initialize QNN backend (load once, reuse for all requests)
        if (!initializeQNN()) {
            LOG_ERROR("Failed to initialize QNN backend");
            return false;
        }

        // Initialize preprocessor (reusable)
        initializePreprocessor();

        // Create Unix domain socket
        if (!initializeSocket()) {
            LOG_ERROR("Failed to initialize socket");
            return false;
        }

        LOG_INFO("VEG Server Combined (New Rope) ready!");
        return true;
    }

    void run() {
        m_running = true;

        while (m_running) {
            LOG_INFO("Waiting for client connection...");

            int clientSocket = accept(m_serverSocket, nullptr, nullptr);
            if (clientSocket < 0) {
                if (m_running) {
                    LOG_ERROR("Accept failed: %s", strerror(errno));
                }
                continue;
            }

            LOG_INFO("Client connected");
            handleClient(clientSocket);
            close(clientSocket);
            LOG_INFO("Client disconnected");
        }
    }

    void stop() {
        m_running = false;
        if (m_serverSocket >= 0) {
            close(m_serverSocket);
            unlink(SOCKET_PATH);
            m_serverSocket = -1;
        }
    }

private:
    // Configuration
    std::string m_backendPath;
    std::string m_systemLibPath;
    std::string m_modelPath;
    int32_t m_targetSize;
    bool m_useOpenCL;

    // Server state
    int m_serverSocket;
    bool m_running;

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

    // Preprocessor (reusable)
    std::unique_ptr<qwen_vl::QwenVLPreprocessor> m_preprocessor;
    qwen_vl::QwenVLConfig m_preprocessorConfig;

    bool initializeSocket() {
        m_serverSocket = socket(AF_UNIX, SOCK_STREAM, 0);
        if (m_serverSocket < 0) {
            LOG_ERROR("Failed to create socket");
            return false;
        }

        // Remove old socket file if exists
        unlink(SOCKET_PATH);

        // Bind socket
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

        if (bind(m_serverSocket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            LOG_ERROR("Failed to bind socket: %s", strerror(errno));
            close(m_serverSocket);
            return false;
        }

        // Listen for connections
        if (listen(m_serverSocket, 5) < 0) {
            LOG_ERROR("Failed to listen: %s", strerror(errno));
            close(m_serverSocket);
            return false;
        }

        LOG_INFO("Server listening on %s", SOCKET_PATH);
        return true;
    }

    void initializePreprocessor() {
        m_preprocessorConfig.patch_size = 16;           // Qwen3-VL: 16x16 patches
        m_preprocessorConfig.temporal_patch_size = 2;
        m_preprocessorConfig.merge_size = 2;
        m_preprocessorConfig.min_image_tokens = 4;
        m_preprocessorConfig.max_image_tokens = 16384;
        m_preprocessorConfig.vit_pos_emb_dim = 32;      // Qwen3-VL: h+w only (32 dims)
        m_preprocessorConfig.image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
        m_preprocessorConfig.image_std = {0.26862954f, 0.26130258f, 0.27577711f};

        m_preprocessor = std::make_unique<qwen_vl::QwenVLPreprocessor>(m_preprocessorConfig, m_useOpenCL);
        LOG_INFO("Preprocessor initialized (OpenCL: %s)", m_useOpenCL ? "enabled" : "disabled");
    }

    bool initializeQNN() {
        LOG_INFO("Initializing QNN backend...");

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

        LOG_INFO("QNN backend initialized successfully");
        return true;
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

        QnnHtpDevice_CustomConfig_t htpDevConfig;
        htpDevConfig.option   = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
        htpDevConfig.socModel = 72;  // SM8650 / Snapdragon 8 Gen 3

        QnnDevice_Config_t devConfig;
        devConfig.option       = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        devConfig.customConfig = &htpDevConfig;

        const QnnDevice_Config_t* devConfigs[] = {&devConfig, nullptr};

        Qnn_ErrorHandle_t error = m_qnnInterface.QNN_INTERFACE_VER_NAME.deviceCreate(
            nullptr, devConfigs, &m_deviceHandle);
        if (error != QNN_SUCCESS) {
            LOG_WARN("deviceCreate returned %lu — continuing with null device handle", (unsigned long)error);
            m_deviceHandle = nullptr;
        } else {
            LOG_INFO("Device created successfully");
        }
        return true;
    }

    bool loadContextFromBinary() {
        LOG_INFO("Loading context from binary: %s", m_modelPath.c_str());

        std::ifstream file(m_modelPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open binary file: %s", m_modelPath.c_str());
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

        error = m_qnnInterface.QNN_INTERFACE_VER_NAME.contextCreateFromBinary(
            m_backendHandle,
            m_deviceHandle,
            nullptr,
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

    void handleClient(int clientSocket) {
        // Read image path length
        uint32_t pathLen;
        if (recv(clientSocket, &pathLen, sizeof(pathLen), MSG_WAITALL) != sizeof(pathLen)) {
            LOG_ERROR("Failed to read path length");
            sendError(clientSocket);
            return;
        }

        // Read image path
        std::vector<char> pathBuf(pathLen + 1, 0);
        if (recv(clientSocket, pathBuf.data(), pathLen, MSG_WAITALL) != (ssize_t)pathLen) {
            LOG_ERROR("Failed to read image path");
            sendError(clientSocket);
            return;
        }

        std::string imagePath(pathBuf.data(), pathLen);
        LOG_INFO("Processing image: %s", imagePath.c_str());

        // Validate image path exists
        struct stat pathStat;
        if (stat(imagePath.c_str(), &pathStat) != 0) {
            LOG_ERROR("Image file does not exist: %s", imagePath.c_str());
            sendError(clientSocket);
            return;
        }

        if (!S_ISREG(pathStat.st_mode)) {
            LOG_ERROR("Path is not a regular file: %s", imagePath.c_str());
            sendError(clientSocket);
            return;
        }

        // Process image and get embeddings
        std::vector<float> embeddings;
        if (!processImage(imagePath, embeddings)) {
            LOG_ERROR("Failed to process image");
            sendError(clientSocket);
            return;
        }

        // Send success + embeddings
        uint32_t status = 0; // 0 = success
        uint32_t numFloats = embeddings.size();

        send(clientSocket, &status, sizeof(status), 0);
        send(clientSocket, &numFloats, sizeof(numFloats), 0);
        send(clientSocket, embeddings.data(), numFloats * sizeof(float), 0);

        LOG_INFO("Sent %u floats to client", numFloats);
    }

    void sendError(int clientSocket) {
        uint32_t status = 1; // 1 = error
        uint32_t numFloats = 0;
        send(clientSocket, &status, sizeof(status), 0);
        send(clientSocket, &numFloats, sizeof(numFloats), 0);
    }

    bool processImage(const std::string& imagePath, std::vector<float>& embeddings) {
        try {
            auto totalStart = std::chrono::high_resolution_clock::now();

            // Stage 1: Preprocess image
            LOG_INFO("Stage 1: Image preprocessing...");
            auto preprocessStart = std::chrono::high_resolution_clock::now();

            qwen_vl::PreprocessedData preprocessed = m_preprocessor->preprocessImage(imagePath, m_targetSize);

            // Validate preprocessing succeeded
            if (preprocessed.seq_len == 0 || preprocessed.pixel_values.empty()) {
                LOG_ERROR("Preprocessing failed - empty output");
                return false;
            }

            if (preprocessed.image_grid_thw.size() < 3) {
                LOG_ERROR("Preprocessing failed - invalid grid_thw");
                return false;
            }

            auto preprocessEnd = std::chrono::high_resolution_clock::now();
            auto preprocessDuration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocessEnd - preprocessStart);
            LOG_INFO("  Preprocessing: %ld ms (seq_len=%d, grid=[%d,%d,%d])",
                     preprocessDuration.count(), preprocessed.seq_len,
                     preprocessed.image_grid_thw[0], preprocessed.image_grid_thw[1], preprocessed.image_grid_thw[2]);

        // Stage 2: QNN inference
        LOG_INFO("Stage 2: QNN inference...");
        auto inferenceStart = std::chrono::high_resolution_clock::now();

        if (m_graphsCount == 0) {
            LOG_ERROR("No graphs loaded");
            return false;
        }

        uint32_t graphIdx = 0;
        GraphInfo_t& graphInfo = m_graphInfo[graphIdx];

        // Prepare input tensors
        Qnn_Tensor_t* inputs = nullptr;
        if (!prepareInputTensors(&inputs, graphIdx, preprocessed)) {
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
        LOG_INFO("  Inference: %ld ms", inferenceDuration.count());

        // Extract embeddings from output
        if (!extractEmbeddings(outputs, graphInfo.numOutputTensors, embeddings)) {
            LOG_ERROR("Failed to extract embeddings");
            tearDownTensors(inputs, graphInfo.numInputTensors);
            tearDownTensors(outputs, graphInfo.numOutputTensors);
            return false;
        }

        // Cleanup tensors
        tearDownTensors(inputs, graphInfo.numInputTensors);
        tearDownTensors(outputs, graphInfo.numOutputTensors);

        auto totalEnd = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);
        LOG_INFO("Total processing time: %ld ms", totalDuration.count());

        return true;

        } catch (const std::exception& e) {
            LOG_ERROR("Exception during image processing: %s", e.what());
            return false;
        } catch (...) {
            LOG_ERROR("Unknown exception during image processing");
            return false;
        }
    }

    bool prepareInputTensors(Qnn_Tensor_t** inputs, uint32_t graphIdx,
                             const qwen_vl::PreprocessedData& preprocessed) {
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

        // Map tensor names to preprocessed data (Qwen3-VL VEG has 3 inputs)
        // Support both naming conventions: preprocessor output and model graph names
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
            } else if (dataType == QNN_DATATYPE_FLOAT_16) {
                uint16_t* fp16Data = reinterpret_cast<uint16_t*>(data);
                for (size_t j = 0; j < srcData->size(); j++) {
                    fp16Data[j] = float32ToFloat16((*srcData)[j]);
                }
            } else if (dataType == QNN_DATATYPE_FLOAT_32) {
                memcpy(data, srcData->data(), srcData->size() * sizeof(float));
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
        }

        return true;
    }

    bool extractEmbeddings(Qnn_Tensor_t* outputs, uint32_t numOutputs, std::vector<float>& embeddings) {
        // Find the vision_embeddings output (usually the first/only output)
        for (uint32_t i = 0; i < numOutputs; i++) {
            const Qnn_Tensor_t& tensor = outputs[i];
            Qnn_ClientBuffer_t clientBuf = QNN_TENSOR_GET_CLIENT_BUF(&tensor);
            Qnn_DataType_t dataType = QNN_TENSOR_GET_DATA_TYPE(&tensor);

            if (dataType == QNN_DATATYPE_UFIXED_POINT_16 || dataType == QNN_DATATYPE_SFIXED_POINT_16) {
                size_t numElements = clientBuf.dataSize / sizeof(uint16_t);
                embeddings.resize(numElements);
                const uint16_t* quantizedData = reinterpret_cast<const uint16_t*>(clientBuf.data);

                Qnn_QuantizeParams_t qParams = QNN_TENSOR_GET_QUANT_PARAMS(&tensor);
                float scale = 1.0f;
                int32_t offset = 0;

                if (qParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                    scale = qParams.scaleOffsetEncoding.scale;
                    offset = qParams.scaleOffsetEncoding.offset;
                }

                for (size_t j = 0; j < numElements; j++) {
                    embeddings[j] = dequantizeFromUint16(quantizedData[j], scale, offset);
                }
            } else if (dataType == QNN_DATATYPE_FLOAT_16) {
                size_t numElements = clientBuf.dataSize / sizeof(uint16_t);
                embeddings.resize(numElements);
                const uint16_t* fp16Data = reinterpret_cast<const uint16_t*>(clientBuf.data);

                for (size_t j = 0; j < numElements; j++) {
                    embeddings[j] = float16ToFloat32(fp16Data[j]);
                }
            } else if (dataType == QNN_DATATYPE_FLOAT_32) {
                size_t numElements = clientBuf.dataSize / sizeof(float);
                embeddings.resize(numElements);
                memcpy(embeddings.data(), clientBuf.data, clientBuf.dataSize);
            } else {
                LOG_WARN("Unknown data type for output tensor, copying raw bytes");
                embeddings.resize(clientBuf.dataSize / sizeof(float));
                memcpy(embeddings.data(), clientBuf.data, clientBuf.dataSize);
            }

            // Only process first output tensor
            break;
        }

        LOG_INFO("Extracted %zu embedding values", embeddings.size());
        return !embeddings.empty();
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

    void cleanupQNN() {
        LOG_INFO("Cleaning up QNN resources...");

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

// Global server instance for signal handler
VEGServerCombined* g_server = nullptr;

void signalHandler(int sig) {
    LOG_INFO("Received signal %d, shutting down...", sig);
    if (g_server) {
        g_server->stop();
    }
}

void showHelp(const char* progName) {
    std::cout << "\nVEG Server Combined (New Rope) - Unified VEG inference server\n";
    std::cout << "\nUSAGE:\n";
    std::cout << "  " << progName << " [OPTIONS]\n\n";
    std::cout << "REQUIRED OPTIONS:\n";
    std::cout << "  --backend <FILE>          Path to QNN backend library (libQnnHtp.so)\n";
    std::cout << "  --system_library <FILE>   Path to QNN system library (libQnnSystem.so)\n";
    std::cout << "  --model <FILE>            Path to serialized binary (veg.serialized.bin)\n";
    std::cout << "\nOPTIONAL OPTIONS:\n";
    std::cout << "  --target_size <SIZE>      Target image size (default: 448)\n";
    std::cout << "  --no_opencl               Disable OpenCL acceleration\n";
    std::cout << "  --help                    Show this help message\n";
    std::cout << "\nPROTOCOL:\n";
    std::cout << "  Client -> Server: [4 bytes: path_len][path_len bytes: image_path]\n";
    std::cout << "  Server -> Client: [4 bytes: status][4 bytes: num_floats][num_floats * 4 bytes: embeddings]\n";
    std::cout << "\nSOCKET PATH: " << SOCKET_PATH << "\n";
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    std::string backendPath;
    std::string systemLibPath;
    std::string modelPath;
    int32_t targetSize = 448;
    bool useOpenCL = true;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            showHelp(argv[0]);
            return 0;
        } else if (arg == "--backend" && i + 1 < argc) {
            backendPath = argv[++i];
        } else if (arg == "--system_library" && i + 1 < argc) {
            systemLibPath = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--target_size" && i + 1 < argc) {
            targetSize = std::stoi(argv[++i]);
        } else if (arg == "--no_opencl") {
            useOpenCL = false;
        } else {
            LOG_ERROR("Unknown argument: %s", arg.c_str());
            showHelp(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (backendPath.empty() || systemLibPath.empty() || modelPath.empty()) {
        LOG_ERROR("Missing required arguments");
        showHelp(argv[0]);
        return 1;
    }

    // Setup signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    LOG_INFO("========================================");
    LOG_INFO("  VEG Server Combined (New Rope)");
    LOG_INFO("========================================");
    LOG_INFO("Backend: %s", backendPath.c_str());
    LOG_INFO("System Library: %s", systemLibPath.c_str());
    LOG_INFO("Model: %s", modelPath.c_str());
    LOG_INFO("Target Size: %d", targetSize);
    LOG_INFO("OpenCL: %s", useOpenCL ? "Enabled" : "Disabled");
    LOG_INFO("========================================\n");

    VEGServerCombined server(backendPath, systemLibPath, modelPath, targetSize, useOpenCL);
    g_server = &server;

    if (!server.initialize()) {
        LOG_ERROR("Server initialization failed");
        return 1;
    }

    LOG_INFO("========================================");
    LOG_INFO("  VEG Server Started");
    LOG_INFO("  Socket: %s", SOCKET_PATH);
    LOG_INFO("========================================");

    server.run();

    LOG_INFO("Server stopped");
    return 0;
}
