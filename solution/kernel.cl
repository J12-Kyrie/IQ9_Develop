// ============================================================
// kernel.cl — transpose_to_patch (exp_7: reordered loops for coalescing)
// Strategy: Loop order (t, ph, pw, c) — innermost c reads consecutive addresses
// Direct index computation for correct output layout
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

    int tIdx = seqIdx / (merged_gridH * merged_gridW * merge_size * merge_size);
    int remainder = seqIdx % (merged_gridH * merged_gridW * merge_size * merge_size);
    int hIdx = remainder / (merged_gridW * merge_size * merge_size);
    remainder = remainder % (merged_gridW * merge_size * merge_size);
    int wIdx = remainder / (merge_size * merge_size);
    remainder = remainder % (merge_size * merge_size);
    int mergeH = remainder / merge_size;
    int mergeW = remainder % merge_size;

    int tps = temporal_patch_size;
    int pp = patch_size * patch_size;
    int tpp = tps * pp;
    int baseH = hIdx * merge_size * patch_size + mergeH * patch_size;
    int baseW = wIdx * merge_size * patch_size + mergeW * patch_size;
    int baseT = tIdx * tps;
    int HW_C = H * W * C;
    int WC = W * C;
    int outBase = seqIdx * input_dim;

    for (int t = 0; t < tps; t++) {
        int tBase = (baseT + t) * HW_C;
        for (int ph = 0; ph < patch_size; ph++) {
            int rowBase = tBase + (baseH + ph) * WC + baseW * C;
            for (int pw = 0; pw < patch_size; pw++) {
                int inAddr = rowBase + pw * C;
                int elemIdx_base = t * pp + ph * patch_size + pw;
                output[outBase + elemIdx_base]         = input[inAddr];
                output[outBase + elemIdx_base + tpp]   = input[inAddr + 1];
                output[outBase + elemIdx_base + 2*tpp] = input[inAddr + 2];
            }
        }
    }
}
