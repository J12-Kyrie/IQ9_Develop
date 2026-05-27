// ============================================================
// kernel.cl — transpose_to_patch (exp_3: nested loops + strength reduction)
// Strategy: 3-level nested loop (c, t, hw) eliminates divisions in inner loop
// All divisors are runtime values to avoid Adreno compiler bug
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

    int patch_area = patch_size * patch_size;
    int tps = temporal_patch_size;
    int baseH = hIdx * merge_size * patch_size + mergeH * patch_size;
    int baseW = wIdx * merge_size * patch_size + mergeW * patch_size;
    int baseT = tIdx * tps;
    int HW = H * W;
    int WC = W * C;
    int HW_C = HW * C;
    int outBase = seqIdx * input_dim;

    // 3-level nested loop: c * tps * patch_area = input_dim
    // Inner loop has ZERO divisions — just increment and compare
    int elemIdx = 0;
    for (int c = 0; c < C; c++) {
        int srcC = c;
        int cBase = srcC;
        for (int t = 0; t < tps; t++) {
            int srcT = baseT + t;
            int tBase = srcT * HW_C + cBase;
            for (int ph = 0; ph < patch_size; ph++) {
                int srcH = baseH + ph;
                int rowBase = tBase + srcH * WC;
                for (int pw = 0; pw < patch_size; pw++) {
                    int srcW = baseW + pw;
                    output[outBase + elemIdx] = input[rowBase + srcW * C];
                    elemIdx++;
                }
            }
        }
    }
}
