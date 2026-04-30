#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>

static bool ValidatePpm(const char* path, int expected_w, int expected_h) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "FAIL: cannot open %s\n", path);
        return false;
    }

    char magic[3] = {};
    int w = 0, h = 0, maxval = 0;
    if (std::fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxval) != 4) {
        std::fprintf(stderr, "FAIL: %s bad header\n", path);
        std::fclose(f);
        return false;
    }

    if (std::strcmp(magic, "P6") != 0) {
        std::fprintf(stderr, "FAIL: %s not P6 (got %s)\n", path, magic);
        std::fclose(f);
        return false;
    }

    if (expected_w > 0 && w != expected_w) {
        std::fprintf(stderr, "FAIL: %s width %d != expected %d\n", path, w, expected_w);
        std::fclose(f);
        return false;
    }

    if (expected_h > 0 && h != expected_h) {
        std::fprintf(stderr, "FAIL: %s height %d != expected %d\n", path, h, expected_h);
        std::fclose(f);
        return false;
    }

    if (maxval != 255) {
        std::fprintf(stderr, "FAIL: %s maxval %d != 255\n", path, maxval);
        std::fclose(f);
        return false;
    }

    // Skip single whitespace after maxval
    std::fgetc(f);

    size_t pixel_count = static_cast<size_t>(w) * static_cast<size_t>(h);
    size_t data_size = pixel_count * 3;
    uint8_t* data = static_cast<uint8_t*>(std::malloc(data_size));
    if (!data) {
        std::fprintf(stderr, "FAIL: %s malloc failed\n", path);
        std::fclose(f);
        return false;
    }

    size_t read = std::fread(data, 1, data_size, f);
    std::fclose(f);

    if (read != data_size) {
        std::fprintf(stderr, "FAIL: %s short data: %zu / %zu\n", path, read, data_size);
        std::free(data);
        return false;
    }

    // Statistics
    uint64_t sum = 0;
    uint8_t min_val = 255, max_val = 0;
    bool all_zero = true;

    for (size_t i = 0; i < data_size; ++i) {
        uint8_t v = data[i];
        sum += v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        if (v != 0) all_zero = false;
    }

    std::free(data);

    double avg = static_cast<double>(sum) / static_cast<double>(data_size);

    if (all_zero) {
        std::fprintf(stderr, "FAIL: %s all pixels are zero\n", path);
        return false;
    }

    std::fprintf(stdout, "  OK: %s  %dx%d  min=%d max=%d avg=%.1f\n",
                 path, w, h, min_val, max_val, avg);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <ppm_dir> [expected_w] [expected_h]\n", argv[0]);
        return 1;
    }

    const char* dir_path = argv[1];
    int expected_w = (argc >= 3) ? std::atoi(argv[2]) : 0;
    int expected_h = (argc >= 4) ? std::atoi(argv[3]) : 0;

    DIR* dir = opendir(dir_path);
    if (!dir) {
        std::fprintf(stderr, "cannot open directory: %s\n", dir_path);
        return 1;
    }

    int pass = 0, fail = 0, total = 0;
    struct dirent* entry;

    while ((entry = readdir(dir)) != nullptr) {
        size_t len = std::strlen(entry->d_name);
        if (len < 4 || std::strcmp(entry->d_name + len - 4, ".ppm") != 0) {
            continue;
        }

        char path[1024];
        std::snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

        ++total;
        if (ValidatePpm(path, expected_w, expected_h)) {
            ++pass;
        } else {
            ++fail;
        }
    }

    closedir(dir);

    std::fprintf(stdout, "\n=== PPM Validation: %d passed, %d failed, %d total ===\n",
                 pass, fail, total);

    if (total == 0) {
        std::fprintf(stderr, "FAIL: no PPM files found in %s\n", dir_path);
        return 1;
    }

    return (fail > 0) ? 1 : 0;
}
