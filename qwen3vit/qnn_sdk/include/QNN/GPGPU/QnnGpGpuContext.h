//==============================================================================
//
// Copyright (c) 2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  A header which defines the QNN GPGPU specialization of the QnnContext.h interface.
 */

#ifndef QNN_GPGPU_CONTEXT_H
#define QNN_GPGPU_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A struct which defines the QNN GPGPU context custom configuration options.
 *        Objects of this type are to be referenced through QnnContext_CustomConfig_t.
 */

typedef struct {
    const char *trtCompiledModelPath;
    const char *trtLibPath;
    const char *qnnRpcHostIp;
    const char *contextBinaryPath;
    const char *inputListPath;
} QnnGpGpuContext_CustomConfig_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
