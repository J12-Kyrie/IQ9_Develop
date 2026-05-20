//=============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All rights reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN HTP MCP Common components
 *
 *         This file defines versioning and other identification details
 *         and supplements QnnCommon.h for HTP MCP backend
 */

#ifndef QNN_HTP_MCP_COMMON_H
#define QNN_HTP_MCP_COMMON_H

#include "QnnCommon.h"

/// HTP MCP Backend identifier
#define QNN_BACKEND_ID_HTP_MCP 11

/// HTP MCP interface provider
#define QNN_HTP_MCP_INTERFACE_PROVIDER_NAME "HTP_MCP_QTI_AISW"

// HTP MCP API Version values
#define QNN_HTP_MCP_API_VERSION_MAJOR 2
#define QNN_HTP_MCP_API_VERSION_MINOR 9
#define QNN_HTP_MCP_API_VERSION_PATCH 0

// clang-format off

/// Macro to set Qnn_ApiVersion_t for HTP MCP backend
#define QNN_HTP_MCP_API_VERSION_INIT                             \
  {                                                              \
    {                                                            \
        QNN_API_VERSION_MAJOR,        /*coreApiVersion.major*/   \
        QNN_API_VERSION_MINOR,        /*coreApiVersion.major*/   \
        QNN_API_VERSION_PATCH         /*coreApiVersion.major*/   \
    },                                                           \
    {                                                            \
      QNN_HTP_MCP_API_VERSION_MAJOR, /*backendApiVersion.major*/ \
      QNN_HTP_MCP_API_VERSION_MINOR, /*backendApiVersion.minor*/ \
      QNN_HTP_MCP_API_VERSION_PATCH  /*backendApiVersion.patch*/ \
    }                                                            \
  }

// clang-format on

// HTP MCP Context blob Version values
#define QNN_HTP_MCP_CONTEXT_BLOB_VERSION_MAJOR 3
#define QNN_HTP_MCP_CONTEXT_BLOB_VERSION_MINOR 3
#define QNN_HTP_MCP_CONTEXT_BLOB_VERSION_PATCH 2

#endif  // QNN_HTP_MCP_COMMON_H
