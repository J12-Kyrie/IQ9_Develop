// cpu_baseline.cpp — CPU reference for transpose_to_patch
#include <vector>

void cpu_baseline(const std::vector<float>& input, std::vector<float>& output, int total_elements) {
    const int T = 2;
    const int C = 3;
    const int patch_size = 16;
    const int merge_size = 2;
    const int temporal_patch_size = 2;
    const int input_dim = 1536;
    const int patch_area = patch_size * patch_size;
    const int tps_pp = temporal_patch_size * patch_area;

    // Infer H, W from total_elements = T * H * W * C
    int HW = total_elements / (T * C);
    int H = 0, W = 0;
    for (int h = 16; h <= 2048; h += 16) {
        if (HW % h == 0) {
            int w = HW / h;
            if (h == w && (h / patch_size) % merge_size == 0) {
                H = h; W = w;
                break;
            }
        }
    }
    if (H == 0) return;

    int gridH = H / patch_size;
    int gridW = W / patch_size;
    int merged_gridH = gridH / merge_size;
    int merged_gridW = gridW / merge_size;
    int gridT = T / temporal_patch_size;
    int seqLength = gridT * merged_gridH * merged_gridW * merge_size * merge_size;

    output.resize((size_t)seqLength * input_dim);

    for (int seqIdx = 0; seqIdx < seqLength; seqIdx++) {
        int tIdx = seqIdx / (merged_gridH * merged_gridW * merge_size * merge_size);
        int remainder = seqIdx % (merged_gridH * merged_gridW * merge_size * merge_size);
        int hIdx = remainder / (merged_gridW * merge_size * merge_size);
        remainder = remainder % (merged_gridW * merge_size * merge_size);
        int wIdx = remainder / (merge_size * merge_size);
        remainder = remainder % (merge_size * merge_size);
        int mergeH = remainder / merge_size;
        int mergeW = remainder % merge_size;

        int baseH = hIdx * merge_size * patch_size + mergeH * patch_size;
        int baseW = wIdx * merge_size * patch_size + mergeW * patch_size;
        int baseT = tIdx * temporal_patch_size;
        int HWx = H * W;
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

            int srcIdx = srcT * HWx * C + srcH * WC + srcW * C + cIdx;
            int dstIdx = seqIdx * input_dim + elemIdx;

            output[dstIdx] = input[srcIdx];
        }
    }
}
