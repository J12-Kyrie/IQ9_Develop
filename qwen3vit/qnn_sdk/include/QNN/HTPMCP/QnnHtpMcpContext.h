//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief QNN HTP MCP component Context API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnContext.h for HTP MCP backend
 */

#ifndef QNN_HTP_MCP_CONTEXT_H
#define QNN_HTP_MCP_CONTEXT_H

#include "QnnContext.h"

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
 * @brief This enum provides different HTP MCP context mode
 *        options that can be used to configure the context mode
 *        In DEFAULT mode, the backend manages enabling context automatically
 *        during QnnContext_CreateFromBinary.
 *        In MANUAL enable mode, the application has full control on enabling/disabling
 *        the context on the device as needed.
 */
typedef enum {
  QNN_HTP_MCP_CONTEXT_MODE_DEFAULT       = 1,
  QNN_HTP_MCP_CONTEXT_MODE_MANUAL_ENABLE = 2,
  QNN_HTP_MCP_CONTEXT_MODE_UNKNOWN       = 0x7fffffff
} QnnHtpMcpContext_Mode_t;

/**
 * @brief This enum provides different HTP MCP state
 *        options that can be used to enable or disable the context
 */
typedef enum {
  QNN_HTP_MCP_CONTEXT_STATE_ENABLE_CONTEXT  = 1,
  QNN_HTP_MCP_CONTEXT_STATE_DISABLE_CONTEXT = 2,
  QNN_HTP_MCP_CONTEXT_STATE_UNKNOWN         = 0x7fffffff
} QnnHtpMcpContext_State_t;

/**
 * @brief This enum provides different MCP context configuration
 *        options associated with QnnContext
 */
typedef enum {
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_MODE                    = 1,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_STATE                   = 2,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_HEAP_SIZE               = 3,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_ELF_PATH                = 4,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_RETRY_TIMEOUT_MS        = 5,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_RETRIES                 = 6,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_RETIRED_1               = 7,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_COMBINED_IO_DMA_ENABLED = 8,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_CRC_ENABLED             = 9,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_CRC_START_BLOCK_SIZE    = 10,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_CRC_END_BLOCK_SIZE      = 11,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_CRC_STRIDE_INTERVAL     = 12,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_CRC_STRIDE_SIZE         = 13,
  QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_UNKNOWN                 = 0x7fffffff
} QnnHtpMcpContext_ConfigOption_t;

//=============================================================================
// Public Functions
//=============================================================================

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------

// clang-format off

/**
 * @brief        Structure describing the set of configurations supported by context.
 *               Objects of this type are to be referenced through QnnContext_CustomConfig_t.
 *
 *               The struct has two fields - option and a union of corresponding config values
 *               Based on the option corresponding item in the union can be used to specify
 *               config.
 *
 *               Below is the Map between QnnHtpMcpContext_CustomConfig_t and config value
 *
 *               \verbatim embed:rst:leading-asterisk
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | #  | Config Option                                                       | Configuration Struct/value            |
 *               +====+=====================================================================+=======================================+
 *               | 1  | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_MODE                              | QnnHtpMcpContext_Mode_t               |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 2  | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_STATE                             | QnnHtpMcpContext_State_t              |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 3  | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_HEAP_SIZE                         | uint32_t                              |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 4  | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_ELF_PATH                          | const char*                           |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 5  | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_RETRY_TIMEOUT_MS                  | uint32_t                              |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 6  | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_RETRIES                           | uint32_t                              |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 7  | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_RETIRED_1                         | None                                  |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 8  | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_COMBINED_IO_DMA_ENABLED           | bool                                  |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 9  | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_CRC_ENABLED                       | bool                                  |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 10 | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_CRC_START_BLOCK_SIZE              | uint32_t                              |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 11 | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_CRC_END_BLOCK_SIZE                | uint32_t                              |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 12 | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_CRC_STRIDE_INTERVAL               | uint32_t                              |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               | 13 | QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_CRC_STRIDE_SIZE                   | uint32_t                              |
 *               +----+---------------------------------------------------------------------+---------------------------------------+
 *               \endverbatim
 *
 */
typedef struct {
  QnnHtpMcpContext_ConfigOption_t option;
  union {
    // This field sets the context mode which is auto on default
    QnnHtpMcpContext_Mode_t contextMode;
    // This field sets the context state which is unknown on default
    QnnHtpMcpContext_State_t contextState;
    // This field sets the heap size in megabytes for the context
    // Default heap size is 256 MB and can be increased up to 2048 MB
    uint32_t heapSizeMb;
    // This field is used to provide path other than default
    const char* elfPath;
    // This field controls how long (in milliseconds) the MCP BE would wait
    // for model activation/inference completion on each retry.
    uint32_t retryTimeoutMs;
    // This field is used to set the number of times the MCP BE
    // retries upon model activation/inference execution timeout.
    uint32_t retries;
    // This field toggles the Combined IO DMA optimization feature on/off.
    // This feature is enabled by default.
    bool  combinedIODMAEnabled;
    // Enables or disables data-path CRC validation. Default is disabled.
    bool crcEnabled;
    // CRC calculation parameters are relevant only if CRC is enabled.
    // Bytes to include at start of DMA buffer
    uint32_t crcStartBlockSize;
    // Bytes to include at end of DMA buffer
    uint32_t crcEndBlockSize;
    // Include a chunk of data at every interval of DMA buffer
    uint32_t crcStrideInterval;
    // Size of the chunk in bytes
    uint32_t crcStrideSize;
  };
} QnnHtpMcpContext_CustomConfig_t;

/// QnnHtpMcpContext_CustomConfig_t initializer macro
#define QNN_HTP_MCP_CONTEXT_CUSTOM_CONFIG_INIT            \
  {                                                       \
    QNN_HTP_MCP_CONTEXT_CONFIG_OPTION_UNKNOWN, /*option*/ \
    {                                                     \
      QNN_HTP_MCP_CONTEXT_MODE_UNKNOWN /*modeOption*/ \
    }                                                     \
  }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
