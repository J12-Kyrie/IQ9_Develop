// baseline.cl — CPU reference algorithm documentation (read-only)
// This file describes the algorithm for human reference.
// The actual CPU implementation is in cpu_baseline.cpp.

// Algorithm: transpose_to_patch
//
// Input:  float[T][H][W][C] — normalized image (row-major, channels-last)
// Output: float[seq_len][input_dim] — patch embeddings for VIT
//
// Parameters:
//   T=2, H=448, W=448, C=3 (production)
//   patch_size=16, merge_size=2, temporal_patch_size=2
//   input_dim = C * temporal_patch_size * patch_size * patch_size = 1536
//   gridT = T / temporal_patch_size = 1
//   gridH = H / patch_size = 28
//   gridW = W / patch_size = 28
//   merged_gridH = gridH / merge_size = 14
//   merged_gridW = gridW / merge_size = 14
//   seq_len = gridT * merged_gridH * merged_gridW * merge_size * merge_size = 784
//
// For each sequence index seqIdx in [0, seq_len):
//   Decode seqIdx -> (tIdx, hIdx, wIdx, mergeH, mergeW)
//   For each elemIdx in [0, input_dim):
//     Decode elemIdx -> (cIdx, tPatchIdx, patchH, patchW)
//     srcT = tIdx * temporal_patch_size + tPatchIdx
//     srcH = hIdx * merge_size * patch_size + mergeH * patch_size + patchH
//     srcW = wIdx * merge_size * patch_size + mergeW * patch_size + patchW
//     output[seqIdx][elemIdx] = input[srcT][srcH][srcW][cIdx]
//
// Bottleneck: inner loop has 1536 iterations, each with 7 integer division/mod ops
// Adreno 663 has no hardware division unit — this is the performance killer
