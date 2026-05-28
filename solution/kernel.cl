// exp_20c: Manual unroll ph=2 + t=2 (process all 512 positions in one loop)

__kernel void transpose_to_patch(
    __global const float* input, __global float* output,
    const int T, const int H, const int W, const int C,
    const int patch_size, const int merge_size,
    const int temporal_patch_size, const int input_dim)
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
    int ps = patch_size, pp = ps * ps, tpp = 2 * pp;
    int baseH = hIdx * merge_size * ps + mergeH * ps;
    int baseW = wIdx * merge_size * ps + mergeW * ps;
    int baseT = tIdx * 2;
    int WC = W * C, HW_C = H * WC;
    int block_base = baseT * HW_C + baseH * WC + baseW * C;
    int outBase = seqIdx * input_dim;

    // Process t=0 and t=1 with ph unrolled by 2
    for (int t = 0; t < 2; t++) {
        int tBase = block_base + t * HW_C;
        int tOff = t * pp;
        for (int ph = 0; ph < ps; ph += 2) {
            int rb0 = tBase + ph * WC;
            int rb1 = rb0 + WC;
            int re0 = tOff + ph * ps;
            int re1 = re0 + ps;
            #pragma unroll 4
            for (int pw = 0; pw < ps; pw++) {
                float3 c0 = vload3(0, input + rb0 + pw * C);
                float3 c1 = vload3(0, input + rb1 + pw * C);
                output[outBase + re0 + pw]         = c0.x;
                output[outBase + re0 + pw + tpp]   = c0.y;
                output[outBase + re0 + pw + 2*tpp] = c0.z;
                output[outBase + re1 + pw]         = c1.x;
                output[outBase + re1 + pw + tpp]   = c1.y;
                output[outBase + re1 + pw + 2*tpp] = c1.z;
            }
        }
    }
}
