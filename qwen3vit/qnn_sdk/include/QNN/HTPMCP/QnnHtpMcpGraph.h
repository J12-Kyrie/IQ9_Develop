//=============================================================================
//
//  Copyright (c) 2022-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief QNN HTP MCP component Graph API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnGraph.h for HTP MCP backend
 */

#ifndef QNN_HTP_MCP_GRAPH_H
#define QNN_HTP_MCP_GRAPH_H

#include "QnnGraph.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief This enum provides different HTP MCP graph optimization
 *        options that can be used to finalize the graph
 *        for optimum performance.
 */
typedef enum {
  QNN_HTP_MCP_GRAPH_OPTIMIZATION_TYPE_SCHEDULE_THRESHOLD         = 1,
  QNN_HTP_MCP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_RETRIES           = 2,
  QNN_HTP_MCP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG = 3,
  QNN_HTP_MCP_GRAPH_OPTIMIZATION_TYPE_UNKNOWN                    = 0x7fffffff
} QnnHtpMcpGraph_OptimizationType_t;

// clang-format off

/**
 * @brief Struct describing the set of optimization types
 *        and the values associated with each optimization type.
 *
 *        Below is the Map between QnnHtpMcpGraph_OptimizationType_t and allowable values:
 *
 *        \verbatim embed:rst:leading-asterisk
 *        +----+----------------------------------------------------------------+---------------------------------------------------------+
 *        | #  | OptimizationType option                                        | Allowable values                                        |
 *        +====+================================================================+=========================================================+
 *        | 1  | QNN_HTP_MCP_GRAPH_OPTIMIZATION_TYPE_SCHEDULE_THRESHOLD         | Reserved                                                |
 *        +----+----------------------------------------------------------------+---------------------------------------------------------+
 *        | 2  | QNN_HTP_MCP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_RETRIES           | Reserved                                                |
 *        +----+----------------------------------------------------------------+---------------------------------------------------------+
 *        | 3  | QNN_HTP_MCP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG | Defines the optimization strategy used by the HTP MCP   |
 *        |    |                                                                | backend                                                 |
 *        |    |                                                                |                                                         |
 *        |    |                                                                |   1 = Faster preparation time, less optimal graph       |
 *        |    |                                                                |                                                         |
 *        |    |                                                                |   2 = More optimal graph but may take longer to prepare |
 *        +----+----------------------------------------------------------------+---------------------------------------------------------+
 *        \endverbatim
 */
typedef struct {
  QnnHtpMcpGraph_OptimizationType_t type;
  float floatValue;
} QnnHtpMcpGraph_OptimizationOption_t;

/// QnnHtpMcpGraph_OptimizationOption_t initializer macro
#define QNN_HTP_MCP_GRAPH_OPTIMIZATION_OPTION_INIT              \
  {                                                         \
    QNN_HTP_MCP_GRAPH_OPTIMIZATION_TYPE_UNKNOWN, /*type*/       \
    0.0f                                     /*floatValue*/ \
  }
// clang-format on

/**
 * @brief This enum provides different HTP graph configuration
 *        options associated with QnnGraph
 */
typedef enum {
  QNN_HTP_MCP_GRAPH_CONFIG_OPTION_OPTIMIZATION                       = 1,
  QNN_HTP_MCP_GRAPH_CONFIG_OPTION_PRECISION                          = 2,
  QNN_HTP_MCP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB                    = 3,
  QNN_HTP_MCP_GRAPH_CONFIG_OPTION_VTCM_SIZE                          = QNN_HTP_MCP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB,
  QNN_HTP_MCP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF = 4,
  QNN_HTP_MCP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF        = 5,
  QNN_HTP_MCP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS                    = 6,
  QNN_HTP_MCP_GRAPH_CONFIG_OPTION_NUMBER_OF_CORES                    = 7,
  QNN_HTP_MCP_GRAPH_CONFIG_OPTION_PROFILING_LEVEL                    = 8,
  QNN_HTP_MCP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG                    = 9,
  QNN_HTP_MCP_GRAPH_CONFIG_OPTION_UNKNOWN                            = 0x7fffffff
} QnnHtpMcpGraph_ConfigOption_t;

//=============================================================================
// Public Functions
//=============================================================================

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------
// clang-format off
/**
 * @brief A struct for different config parameters in a key value format.
 */
typedef struct {
  const char* key;
  Qnn_Scalar_t value;
} QnnHtpMcpGraph_FinalizeConfig_t;

// clang-format on

// clang-format off

/**
 * @brief        Structure describing the set of configurations supported by graph.
 *               Objects of this type are to be referenced through QnnGraph_CustomConfig_t.
 *
 *               The struct has two fields - option and a union of corresponding config values
 *               Based on the option corresponding item in the union can be used to specify
 *               config.
 *
 *               Below is the Map between QnnHtpMcpGraph_ConfigOption_t and config value
 *
 *               \verbatim embed:rst:leading-asterisk
 *               +----+---------------------------------------------------------------------------------------------+-------------------------------------+
 *               | #  | Config Option                                                                               | Configuration Struct/value          |
 *               +====+=============================================================================================+=====================================+
 *               | 1  | QNN_HTP_MCP_GRAPH_CONFIG_OPTION_OPTIMIZATION                                                | QnnHtpMcpGraph_OptimizationOption_t |
 *               +----+---------------------------------------------------------------------------------------------+-------------------------------------+
 *               | 2  | QNN_HTP_MCP_GRAPH_CONFIG_OPTION_PRECISION                                                   | Qnn_Precision_t                     |
 *               +----+---------------------------------------------------------------------------------------------+-------------------------------------+
 *               | 3  | QNN_HTP_MCP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB/QNN_HTP_MCP_GRAPH_CONFIG_OPTION_VTCM_SIZE   | uint32_t                            |
 *               +----+---------------------------------------------------------------------------------------------+-------------------------------------+
 *               | 4  | QNN_HTP_MCP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF                          | bool                                |
 *               +----+---------------------------------------------------------------------------------------------+-------------------------------------+
 *               | 5  | QNN_HTP_MCP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF                                 | bool                                |
 *               +----+---------------------------------------------------------------------------------------------+-------------------------------------+
 *               | 6  | QNN_HTP_MCP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS                                             | uint64_t                            |
 *               +----+---------------------------------------------------------------------------------------------+-------------------------------------+
 *               | 7  | QNN_HTP_MCP_GRAPH_CONFIG_OPTION_NUMBER_OF_CORES                                             | uint32_t                            |
 *               +----+---------------------------------------------------------------------------------------------+-------------------------------------+
 *               | 8  | QNN_HTP_MCP_GRAPH_CONFIG_OPTION_PROFILING_LEVEL                                             | uint32_t                            |
 *               +----+---------------------------------------------------------------------------------------------+-------------------------------------+
 *               | 9  | QNN_HTP_MCP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG                                             | QnnHtpMcpGraph_FinalizeConfig_t     |
 *               +----+---------------------------------------------------------------------------------------------+-------------------------------------+
 *               \endverbatim
 *
 *               NOTE: Option #6 (i.e. QNN_HTP_MCP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS), can only be
 *               set prior to the first execution of the graph. Proceeding executions will not use
 *               the updated value if user does change it after the first execution.
 */
typedef struct {
  QnnHtpMcpGraph_ConfigOption_t option;
  union {
    QnnHtpMcpGraph_OptimizationOption_t optimizationOption;
    Qnn_Precision_t precision;
    uint32_t vtcmSizeInMB;
    bool foldReluActivationIntoConvOff;
    bool shortDepthConvOnHmxOff;
    uint64_t numHvxThreads;
    uint32_t numCores;
    uint32_t profilingLevel;
    QnnHtpMcpGraph_FinalizeConfig_t finalizeConfig;
  };
} QnnHtpMcpGraph_CustomConfig_t;

// clang-format on
/// QnnHtpMcpGraph_CustomConfig_t initializer macro
#define QNN_HTP_MCP_GRAPH_CUSTOM_CONFIG_INIT                            \
  {                                                                     \
    QNN_HTP_MCP_GRAPH_CONFIG_OPTION_UNKNOWN, /*option*/                 \
    {                                                                   \
      QNN_HTP_MCP_GRAPH_OPTIMIZATION_OPTION_INIT /*optimizationOption*/ \
    }                                                                   \
  }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
