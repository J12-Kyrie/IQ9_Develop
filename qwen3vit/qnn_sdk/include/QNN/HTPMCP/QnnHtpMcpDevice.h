//=============================================================================
//
//  Copyright (c) 2022-2024 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN HTP MCP Device components
 *
 *  This file defines structures and supplements QnnDevice.h for QNN HTP MCP device
 */

#pragma once

#include "QnnCommon.h"
#include "QnnDevice.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

// For deviceType in QnnDevice_HardwareDeviceInfoV1_t
typedef enum {
  QNN_HTP_MCP_DEVICE_TYPE_PCIE    = 0,  // HTP cores are access via PCIe bus
  QNN_HTP_MCP_DEVICE_TYPE_UNKNOWN = 0x7fffffff
} QnnHtpMcpDevice_DeviceType_t;

/**
 * This structure provides info about the device that connect with PCIe bus
 * For online operation, caller should get these info from QnnDevice_getPlatformInfo
 * For offline operation, caller need to create this structure and filling the correct information
 * for QnnDevice_create
 */

/**
 * @brief QNN HTP MCP Device core type
 * This enumeration provides information about the core type inside the SOC.
 */
typedef enum {
  QNN_HTP_MCP_CORE_TYPE_NSP = 0,

  // Valid Supported core types are < QNN_CORE_TYPE_MAX
  QNN_HTP_MCP_CORE_TYPE_MAX,
  QNN_HTP_MCP_CORE_TYPE_UNKNOWN = 0x7fffffff
} QnnHtpMcpDevice_CoreType_t;

typedef struct {
  size_t vtcmSize;   // The VTCM size for each NSP in Mega Byte
                     // The graph option cannot exceed this value
  uint8_t nspTotal;  // Total number of NSPs in this PCIe device
} QnnHtpMcpDevice_PCIeDeviceInfoExtension_t;

/**
 * This structure is being used in QnnDevice_HardwareDeviceInfoV1_t
 * QnnDevice_getPlatformInfo use this structure to list the supported device features/info
 */
typedef struct _QnnDevice_DeviceInfoExtension_t {
  QnnHtpMcpDevice_DeviceType_t devType;
  union UNNAMED {
    QnnHtpMcpDevice_PCIeDeviceInfoExtension_t pcieDevice;
  };
} QnnHtpMcpDevice_DeviceInfoExtension_t;

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif
