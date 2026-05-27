// ============================================================
// kernel.cl — transpose_to_patch (production input_dim=1536)
// Transposes normalized image patches into VIT input format
// Input:  [T, H, W, C] float normalized image
// Output: [seq_len, input_dim] float patch embeddings
// ============================================================

__kernel void transpose_to_patch(
    __global const float* input,
    __global float* output,
    const int T,
    const int H,
    const int W,
    const int C,
    const int patch_size,
    const int merge_size,
    const int temporal_patch_size,
    const int input_dim)
{
    int seqIdx = get_global_id(0);

    int gridT = T / temporal_patch_size;
    int gridH = H / patch_size;
    int gridW = W / patch_size;
    int merged_gridH = gridH / merge_size;
    int merged_gridW = gridW / merge_size;
    int seqLength = gridT * merged_gridH * merged_gridW * merge_size * merge_size;

    if (seqIdx >= seqLength) return;

    // Decode sequence index into (tIdx, hIdx, wIdx, mergeH, mergeW)
    int tIdx = seqIdx / (merged_gridH * merged_gridW * merge_size * merge_size);
    int remainder = seqIdx % (merged_gridH * merged_gridW * merge_size * merge_size);
    int hIdx = remainder / (merged_gridW * merge_size * merge_size);
    remainder = remainder % (merged_gridW * merge_size * merge_size);
    int wIdx = remainder / (merge_size * merge_size);
    remainder = remainder % (merge_size * merge_size);
    int mergeH = remainder / merge_size;
    int mergeW = remainder % merge_size;

    // Precompute strides for inner loop
    int patch_area = patch_size * patch_size;          // 256
    int tps_pp = temporal_patch_size * patch_area;     // 512

    // Precompute base spatial offsets (constant per work-item)
    int baseH = hIdx * merge_size * patch_size + mergeH * patch_size;
    int baseW = wIdx * merge_size * patch_size + mergeW * patch_size;
    int baseT = tIdx * temporal_patch_size;
    int HW = H * W;
    int WC = W * C;

    for (int elemIdx = 0; elemIdx < input_dim; elemIdx++) {
        int cIdx = elemIdx / tps_pp;
        int temp = elemIdx % tps_pp;
        int tPatchIdx = temp / patch_area;
        int patchH = (temp % patch_area) / patch_size;
        int patchW = temp % patch_size;

        int srcT = baseT + tPatchIdx;
        int srcH = baseH + patchH;
        int srcW = baseW + patchW;

        int srcIdx = srcT * HW * C + srcH * WC + srcW * C + cIdx;
        int dstIdx = seqIdx * input_dim + elemIdx;

        output[dstIdx] = input[srcIdx];
    }
}
