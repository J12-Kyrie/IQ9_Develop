// ============================================================
// benchmark.cpp — Generic OpenCL Kernel Benchmark Runner
// Reads kernel.toml for configuration. No domain-specific code.
// Build: cmake ../host && make -j$(nproc)
// Run:   ./benchmark [--quick|--stride N|--full] [--output bench.log]
// ============================================================

#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <map>
#include <functional>

// ---- Error Checking ----
#define CL_CHECK(call, msg) do { \
    cl_int e = (call); \
    if (e != CL_SUCCESS) { fprintf(stderr, "CL Error %d: %s\n", e, msg); return false; } \
} while(0)

// ---- GPU Temperature ----
static float readGpuTemp() {
    const char* zones[] = {
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone5/temp",
    };
    for (const char* z : zones) {
        std::ifstream f(z);
        if (f.is_open()) {
            int temp_mc; f >> temp_mc;
            return temp_mc / 1000.0f;
        }
    }
    return -1.0f;
}

// ---- String Helpers ----
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

static bool startsWith(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

// ---- Minimal TOML Parser (flat + array-of-tables + array-of-dicts) ----
struct KernelArg {
    std::string name;
    std::string type;       // buffer_read, buffer_write, buffer_rw, int, float
    std::string size_expr;  // for buffers
    std::string source;     // "problem_size" or empty
    double literal;         // for int/float with fixed value
    bool has_literal;
};

struct KernelDef {
    std::string function;
    int ndim;
    std::string global_size_expr;
    size_t local_work_size;  // 0 = NULL
    std::vector<KernelArg> args;
};

struct ProblemSize {
    std::map<std::string, double> vars;
};

struct CorrectnessConfig {
    bool enabled;
    std::string mode;  // cpu_baseline, gold_file, none
    double tolerance;
};

struct KernelConfig {
    std::string name;
    std::string description;
    std::vector<KernelDef> kernels;
    std::vector<ProblemSize> quick;
    std::vector<ProblemSize> stride;
    std::vector<ProblemSize> full;
    CorrectnessConfig correctness;
};

// Parse TOML inline table: {key = val, key2 = val2}
static std::map<std::string, double> parseInlineTable(const std::string& s) {
    std::map<std::string, double> m;
    std::string inner = trim(s);
    if (inner.front() == '{') inner = inner.substr(1);
    if (inner.back() == '}') inner.pop_back();
    std::istringstream ss(inner);
    std::string token;
    while (std::getline(ss, token, ',')) {
        size_t eq = token.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(token.substr(0, eq));
        std::string val = trim(token.substr(eq + 1));
        m[key] = std::stod(val);
    }
    return m;
}

// Parse TOML array of inline tables: [{k=v}, {k=v}]
static std::vector<ProblemSize> parseArrayOfTables(const std::string& s) {
    std::vector<ProblemSize> result;
    std::string inner = trim(s);
    // Split by }, {  boundaries
    size_t pos = 0;
    while (pos < inner.size()) {
        size_t start = inner.find('{', pos);
        if (start == std::string::npos) break;
        size_t end = inner.find('}', start);
        if (end == std::string::npos) break;
        std::string table = inner.substr(start, end - start + 1);
        result.push_back({parseInlineTable(table)});
        pos = end + 1;
    }
    return result;
}

static KernelConfig parseKernelToml(const std::string& path) {
    KernelConfig cfg;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s\n", path.c_str());
        return cfg;
    }

    std::string line, section, subsection;
    KernelDef* current_kernel = nullptr;

    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        // Section header
        if (line[0] == '[') {
            if (startsWith(line, "[[kernels")) {
                cfg.kernels.emplace_back();
                current_kernel = &cfg.kernels.back();
                subsection = "kernels";
            } else if (startsWith(line, "[[kernels.args")) {
                if (current_kernel) {
                    current_kernel->args.emplace_back();
                    subsection = "kernels.args";
                }
            } else if (line == "[kernel]") {
                section = "kernel"; subsection = ""; current_kernel = nullptr;
            } else if (line == "[problem_sizes]") {
                section = "problem_sizes"; subsection = ""; current_kernel = nullptr;
            } else if (line == "[correctness]") {
                section = "correctness"; subsection = ""; current_kernel = nullptr;
            }
            continue;
        }

        // Key = value
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        // Remove quotes
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);

        if (section == "kernel") {
            if (key == "name") cfg.name = val;
            else if (key == "description") cfg.description = val;
        } else if (section == "problem_sizes") {
            auto sizes = parseArrayOfTables(val);
            if (key == "quick") cfg.quick = sizes;
            else if (key == "stride") cfg.stride = sizes;
            else if (key == "full") cfg.full = sizes;
        } else if (section == "correctness") {
            if (key == "enabled") cfg.correctness.enabled = (val == "true");
            else if (key == "mode") cfg.correctness.mode = val;
            else if (key == "tolerance") cfg.correctness.tolerance = std::stod(val);
        }

        if (subsection == "kernels" && current_kernel) {
            if (key == "function") current_kernel->function = val;
            else if (key == "ndim") current_kernel->ndim = std::stoi(val);
            else if (key == "global_size") current_kernel->global_size_expr = val;
            else if (key == "local_work_size") current_kernel->local_work_size = std::stoul(val);
        } else if (subsection == "kernels.args" && current_kernel && !current_kernel->args.empty()) {
            auto& arg = current_kernel->args.back();
            if (key == "name") arg.name = val;
            else if (key == "type") arg.type = val;
            else if (key == "size") arg.size_expr = val;
            else if (key == "source") arg.source = val;
            else if (key == "value") { arg.literal = std::stod(val); arg.has_literal = true; }
        }
    }
    return cfg;
}

// ---- Expression Evaluator ----
// Simple evaluator: supports "N", "N * 4", "N * 4 * 3", literal numbers
static double evalExpr(const std::string& expr, const std::map<std::string, double>& vars) {
    std::string e = trim(expr);
    // Try direct variable lookup
    if (vars.count(e)) return vars.at(e);
    // Try literal number
    char* endptr;
    double lit = std::strtod(e.c_str(), &endptr);
    if (*endptr == '\0') return lit;

    // Simple "A * B * C..." chain
    std::istringstream ss(e);
    std::string token;
    double result = 1.0;
    bool expect_op = false;
    while (ss >> token) {
        if (token == "*") { expect_op = false; continue; }
        if (expect_op) { fprintf(stderr, "Unexpected token in expr: %s\n", token.c_str()); return 0; }
        double val;
        if (vars.count(token)) val = vars.at(token);
        else {
            val = std::strtod(token.c_str(), &endptr);
            if (*endptr != '\0') {
                fprintf(stderr, "Unknown variable in expr: %s\n", token.c_str());
                return 0;
            }
        }
        result *= val;
        expect_op = true;
    }
    return result;
}

// ---- OpenCL Init ----
static bool initOpenCL(cl_platform_id* plat, cl_device_id* dev,
                       cl_context* ctx, cl_command_queue* q) {
    cl_platform_id platforms[4]; cl_uint nplat;
    CL_CHECK(clGetPlatformIDs(4, platforms, &nplat), "getPlatformIDs");
    if (nplat == 0) { fprintf(stderr, "No OpenCL platforms\n"); return false; }
    *plat = platforms[0];

    cl_device_id devices[4]; cl_uint ndev;
    CL_CHECK(clGetDeviceIDs(*plat, CL_DEVICE_TYPE_GPU, 4, devices, &ndev), "getDeviceIDs(GPU)");
    if (ndev == 0)
        CL_CHECK(clGetDeviceIDs(*plat, CL_DEVICE_TYPE_DEFAULT, 4, devices, &ndev), "getDeviceIDs(DEFAULT)");
    if (ndev == 0) { fprintf(stderr, "No OpenCL devices\n"); return false; }
    *dev = devices[0];

    char name[256]; size_t wgs;
    clGetDeviceInfo(*dev, CL_DEVICE_NAME, 256, name, NULL);
    clGetDeviceInfo(*dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &wgs, NULL);
    fprintf(stdout, "Device: %s | Max WG: %zu\n", name, wgs);

    cl_int err;
    *ctx = clCreateContext(NULL, 1, dev, NULL, NULL, &err);
    CL_CHECK(err, "createContext");
    cl_queue_properties q_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    *q = clCreateCommandQueueWithProperties(*ctx, *dev, q_props, &err);
    CL_CHECK(err, "createCommandQueue");
    return true;
}

static bool buildProgram(cl_context ctx, cl_device_id dev, cl_program* prog,
                         const char* kernel_path) {
    std::ifstream f(kernel_path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open kernel: %s\n", kernel_path); return false; }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* csrc = src.c_str();
    size_t slen = src.size();
    cl_int err;
    *prog = clCreateProgramWithSource(ctx, 1, &csrc, &slen, &err);
    CL_CHECK(err, "createProgram");
    err = clBuildProgram(*prog, 1, &dev, "-cl-fast-relaxed-math -cl-mad-enable -Werror", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(*prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_sz);
        std::vector<char> log(log_sz + 1);
        clGetProgramBuildInfo(*prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), NULL);
        fprintf(stderr, "Build error:\n%s\n", log.data());
        return false;
    }
    return true;
}

// ---- CPU Baseline (weak symbol — branch provides strong definition) ----
extern void cpu_baseline(const std::vector<float>& input, std::vector<float>& output, int N)
    __attribute__((weak));

// ---- Main ----
int main(int argc, char** argv) {
    // Parse CLI
    const char* kernel_path = KERNEL_PATH;
    const char* toml_path = TOML_PATH;
    int mode = 2;  // 0=quick, 1=stride, 2=full
    const char* output_file = nullptr;
    int custom_stride = 2;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) mode = 0;
        else if (!strcmp(argv[i], "--stride")) { mode = 1; if (i+1 < argc) custom_stride = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--full")) mode = 2;
        else if (!strcmp(argv[i], "--output") && i+1 < argc) output_file = argv[++i];
        else if (!strcmp(argv[i], "--toml") && i+1 < argc) toml_path = argv[++i];
        else if (!strcmp(argv[i], "--kernel") && i+1 < argc) kernel_path = argv[++i];
    }

    // Parse config
    KernelConfig cfg = parseKernelToml(toml_path);
    if (cfg.kernels.empty()) {
        fprintf(stderr, "No [[kernels]] defined in %s\n", toml_path);
        return 1;
    }

    // Select problem sizes for this mode
    std::vector<ProblemSize>* sizes = nullptr;
    if (mode == 0) sizes = &cfg.quick;
    else if (mode == 1) sizes = &cfg.stride;
    else sizes = &cfg.full;

    if (sizes->empty()) {
        fprintf(stderr, "No problem_sizes defined for mode %d\n", mode);
        return 1;
    }

    // Stride filter
    std::vector<ProblemSize> selected;
    if (mode == 1) {
        for (size_t i = 0; i < sizes->size(); i += custom_stride)
            selected.push_back((*sizes)[i]);
    } else {
        selected = *sizes;
    }

    // Init OpenCL
    cl_platform_id plat; cl_device_id dev; cl_context ctx; cl_command_queue queue; cl_program prog;
    if (!initOpenCL(&plat, &dev, &ctx, &queue)) return 1;
    if (!buildProgram(ctx, dev, &prog, kernel_path)) return 1;

    // Create kernels and their arg metadata
    struct KernelInstance {
        cl_kernel kernel;
        std::vector<KernelArg> args;
        int ndim;
        std::string global_size_expr;
        size_t local_work_size;
    };
    std::vector<KernelInstance> instances;

    for (auto& kd : cfg.kernels) {
        cl_int err;
        cl_kernel k = clCreateKernel(prog, kd.function.c_str(), &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to create kernel '%s' (err %d)\n", kd.function.c_str(), err);
            return 1;
        }
        instances.push_back({k, kd.args, kd.ndim, kd.global_size_expr, kd.local_work_size});
    }

    // Open output
    FILE* out = output_file ? fopen(output_file, "w") : stdout;
    fprintf(out, "# OpenCL Kernel Benchmark\n");
    fprintf(out, "# Kernel: %s\n", cfg.name.c_str());
    fprintf(out, "# Timestamp: %lld\n", (long long)std::chrono::system_clock::now().time_since_epoch().count());
    fprintf(out, "# Mode: %s | Problem sizes: %zu\n",
            mode == 0 ? "quick" : (mode == 1 ? "stride" : "full"), selected.size());
    float temp_before = readGpuTemp();
    fprintf(out, "# GPU Temp Before: %.1fC\n", temp_before);

    // Header
    fprintf(out, "%-16s %10s %10s %10s %10s\n",
            "ProblemSize", "Kernel_ms", "Total_ms", "Stdev_ms", "Correct");

    bool all_pass = true;

    for (auto& ps : selected) {
        // Build variable map for expressions
        auto& vars = ps.vars;

        // Describe this problem size for display
        std::string size_label;
        for (auto& [k, v] : vars) {
            if (!size_label.empty()) size_label += "_";
            size_label += k + "=" + std::to_string((int)v);
        }

        // Per-size resources (freed after this iteration)
        struct BufferInfo { cl_mem buf; bool is_output; };
        std::vector<BufferInfo> buffers;
        std::map<std::string, cl_mem> named_buffers;

        // Allocate buffers and set args for each kernel instance
        bool alloc_ok = true;
        for (auto& inst : instances) {
            for (size_t ai = 0; ai < inst.args.size(); ai++) {
                auto& arg = inst.args[ai];
                if (startsWith(arg.type, "buffer")) {
                    size_t bytes = (size_t)evalExpr(arg.size_expr, vars);
                    cl_int cerr;
                    cl_mem_flags flags = CL_MEM_READ_WRITE;
                    if (arg.type == "buffer_read") flags = CL_MEM_READ_ONLY;
                    else if (arg.type == "buffer_write") flags = CL_MEM_WRITE_ONLY;
                    cl_mem buf = clCreateBuffer(ctx, flags, bytes, NULL, &cerr);
                    if (cerr != CL_SUCCESS) {
                        fprintf(stderr, "Buffer alloc failed for %s (%zu bytes): %d\n",
                                arg.name.c_str(), bytes, cerr);
                        alloc_ok = false; break;
                    }
                    bool is_out = (arg.type == "buffer_write" || arg.type == "buffer_rw");
                    buffers.push_back({buf, is_out});
                    named_buffers[arg.name] = buf;
                    clSetKernelArg(inst.kernel, ai, sizeof(cl_mem), &buf);
                } else if (arg.type == "int") {
                    int val;
                    if (arg.source == "problem_size") val = (int)evalExpr("N", vars);
                    else if (arg.has_literal) val = (int)arg.literal;
                    else val = (int)evalExpr(arg.source.empty() ? "0" : arg.source, vars);
                    clSetKernelArg(inst.kernel, ai, sizeof(int), &val);
                } else if (arg.type == "float") {
                    float val;
                    if (arg.has_literal) val = (float)arg.literal;
                    else val = (float)evalExpr(arg.source.empty() ? "0" : arg.source, vars);
                    clSetKernelArg(inst.kernel, ai, sizeof(float), &val);
                }
            }
            if (!alloc_ok) break;
        }
        if (!alloc_ok) { all_pass = false; continue; }

        // Compute global size
        size_t global_size[2] = {1, 1};
        int ndim = instances[0].ndim;
        std::string gs_expr = instances[0].global_size_expr;

        // Parse global_size expression (supports "N" or "[W, H]")
        if (gs_expr.front() == '[') {
            // 2D: [expr1, expr2]
            std::string inner = gs_expr.substr(1, gs_expr.size() - 2);
            size_t comma = inner.find(',');
            global_size[0] = (size_t)evalExpr(trim(inner.substr(0, comma)), vars);
            global_size[1] = (size_t)evalExpr(trim(inner.substr(comma + 1)), vars);
        } else {
            global_size[0] = (size_t)evalExpr(gs_expr, vars);
        }

        size_t* local_ws = (instances[0].local_work_size > 0) ? &instances[0].local_work_size : NULL;

        // Fill input buffers with synthetic data
        for (auto& bi : buffers) {
            if (!bi.is_output) {
                // Fill with gradient pattern based on buffer size
                size_t bytes = 0;
                cl_mem_object_type dummy;
                clGetMemObjectInfo(bi.buf, CL_MEM_SIZE, sizeof(size_t), &bytes, NULL);
                std::vector<unsigned char> pattern(bytes);
                for (size_t i = 0; i < bytes; i++) pattern[i] = (unsigned char)(i & 0xFF);
                clEnqueueWriteBuffer(queue, bi.buf, CL_TRUE, 0, bytes, pattern.data(), 0, NULL, NULL);
            }
        }

        // Warmup
        for (int w = 0; w < 10; w++) {
            for (auto& inst : instances)
                clEnqueueNDRangeKernel(queue, inst.kernel, ndim, NULL, global_size, local_ws, 0, NULL, NULL);
            clFinish(queue);
        }

        // Timed iterations
        const int ITERS = 100;
        double total_kernel = 0, total_all = 0;
        std::vector<double> iter_times; iter_times.reserve(ITERS);

        for (int iter = 0; iter < ITERS; iter++) {
            std::vector<cl_event> events;
            cl_event prev = NULL;

            for (auto& inst : instances) {
                cl_event ev;
                cl_uint num_deps = prev ? 1 : 0;
                cl_event* deps = prev ? &prev : NULL;
                clEnqueueNDRangeKernel(queue, inst.kernel, ndim, NULL, global_size, local_ws,
                                       num_deps, deps, &ev);
                events.push_back(ev);
                prev = ev;
            }

            // Wait for last kernel
            clWaitForEvents(1, &prev);

            double kernel_ms = 0;
            for (auto& ev : events) {
                cl_ulong start, end;
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                kernel_ms += (end - start) / 1e6;
            }

            total_kernel += kernel_ms;
            iter_times.push_back(kernel_ms);
            for (auto& ev : events) clReleaseEvent(ev);
        }

        double avg_kernel = total_kernel / ITERS;
        double sum_sq = 0;
        for (double t : iter_times) { double d = t - avg_kernel; sum_sq += d * d; }
        double stdev = std::sqrt(sum_sq / ITERS);

        // Correctness check
        bool correct = true;
        if (cfg.correctness.enabled && cfg.correctness.mode == "cpu_baseline" && cpu_baseline) {
            // Find first output buffer, read it back
            for (auto& bi : buffers) {
                if (bi.is_output) {
                    size_t bytes = 0;
                    clGetMemObjectInfo(bi.buf, CL_MEM_SIZE, sizeof(size_t), &bytes, NULL);
                    size_t float_count = bytes / sizeof(float);
                    std::vector<float> gpu_out(float_count);
                    clEnqueueReadBuffer(queue, bi.buf, CL_TRUE, 0, bytes, gpu_out.data(), 0, NULL, NULL);

                    // Build CPU input from first input buffer
                    std::vector<float> cpu_in(float_count);
                    for (auto& bi2 : buffers) {
                        if (!bi2.is_output) {
                            size_t in_bytes = 0;
                            clGetMemObjectInfo(bi2.buf, CL_MEM_SIZE, sizeof(size_t), &in_bytes, NULL);
                            std::vector<unsigned char> raw(in_bytes);
                            clEnqueueReadBuffer(queue, bi2.buf, CL_TRUE, 0, in_bytes, raw.data(), 0, NULL, NULL);
                            // Interpret as float if size matches
                            if (in_bytes == bytes) {
                                memcpy(cpu_in.data(), raw.data(), bytes);
                            }
                            break;
                        }
                    }

                    std::vector<float> cpu_out;
                    cpu_baseline(cpu_in, cpu_out, (int)float_count);

                    float max_diff = 0;
                    for (size_t i = 0; i < float_count; i++) {
                        float d = fabsf(gpu_out[i] - cpu_out[i]);
                        if (d > max_diff) max_diff = d;
                    }
                    correct = (max_diff < cfg.correctness.tolerance);
                    if (!correct) {
                        fprintf(stderr, "Correctness FAIL: max_diff=%.6f (tolerance=%.6f)\n",
                                max_diff, cfg.correctness.tolerance);
                    }
                    break;
                }
            }
        }

        if (!correct) all_pass = false;

        fprintf(out, "%-16s %10.3f %10.3f %10.3f %10s\n",
                size_label.c_str(), avg_kernel, avg_kernel, stdev,
                correct ? "PASS" : "FAIL");

        // Cleanup per-size buffers
        for (auto& bi : buffers) clReleaseMemObject(bi.buf);
    }

    float temp_after = readGpuTemp();
    fprintf(out, "# GPU Temp After: %.1fC (delta: +%.1fC)\n",
            temp_after, (temp_before > 0 && temp_after > 0) ? (temp_after - temp_before) : 0.0f);
    fprintf(out, "# Overall: %s\n", all_pass ? "PASS" : "FAIL");

    if (output_file) { fclose(out); fprintf(stdout, "Results written to %s\n", output_file); }

    for (auto& inst : instances) clReleaseKernel(inst.kernel);
    clReleaseProgram(prog); clReleaseCommandQueue(queue); clReleaseContext(ctx);
    return all_pass ? 0 : 1;
}
