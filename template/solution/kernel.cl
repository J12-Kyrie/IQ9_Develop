// ============================================================
// kernel.cl — [REPLACE_WITH_KERNEL_NAME]
// Agent-editable file for /optimize loop
// ============================================================

__kernel void hello_kernel(
    __global const float* input,
    __global float* output,
    const int N)
{
    int idx = get_global_id(0);
    if (idx >= N) return;
    output[idx] = input[idx] * 2.0f;
}
