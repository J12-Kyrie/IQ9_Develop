//=============================================================================
//
//  Copyright (c) 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN GPGPU Common components
 *
 *         This file defines versioning and other identification details
 *         and supplements QnnCommon.h for GPGPU backend
 */

#ifndef QNN_GPGPU_COMMON_H
#define QNN_GPGPU_COMMON_H

#include "QnnCommon.h"

/// GPGPU Backend identifier
#define QNN_BACKEND_ID_GPGPU 10

/// GPGPU interface provider
#define QNN_GPGPU_INTERFACE_PROVIDER_NAME "GPGPU_QTI_AISW"

// GPGPU API Version values
#define QNN_GPGPU_API_VERSION_MAJOR 2
#define QNN_GPGPU_API_VERSION_MINOR 0
#define QNN_GPGPU_API_VERSION_PATCH 0

// clang-format off

/// Macro to set Qnn_ApiVersion_t for GPGPU backend
#define QNN_GPGPU_API_VERSION_INIT                                 \
  {                                                              \
    {                                                            \
      QNN_API_VERSION_MAJOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_MINOR,     /*coreApiVersion.major*/        \
      QNN_API_VERSION_PATCH      /*coreApiVersion.major*/        \
    },                                                           \
    {                                                            \
      QNN_GPGPU_API_VERSION_MAJOR, /*backendApiVersion.major*/     \
      QNN_GPGPU_API_VERSION_MINOR, /*backendApiVersion.minor*/     \
      QNN_GPGPU_API_VERSION_PATCH  /*backendApiVersion.patch*/     \
    }                                                            \
  }
// clang-format on

#endif  // QNN_GPGPU_COMMON_H
