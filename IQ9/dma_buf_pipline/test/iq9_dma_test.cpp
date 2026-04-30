#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <cstring>

// 严格按照 1920 * 1080 * 4 RGBA
constexpr size_t FRAME_WIDTH = 1920;
constexpr size_t FRAME_HEIGHT = 1080;
constexpr size_t FRAME_CHANNELS = 4;
constexpr size_t SLOT_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS; 
constexpr int SLOT_COUNT = 8;
constexpr size_t TOTAL_SIZE = SLOT_SIZE * SLOT_COUNT;

constexpr const char* HEAP_PATH = "/dev/dma_heap/qcom,system";

inline long long now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int alloc_dma_buf(size_t size) {
    int heap_fd = open(HEAP_PATH, O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) return -1;
    struct dma_heap_allocation_data alloc_data = {};
    alloc_data.len = size;
    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
    ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data);
    close(heap_fd);
    return alloc_data.fd;
}

void test_memory_bandwidth(void* ptr, size_t size) {
    std::cout << "\n======================================================\n";
    std::cout << " 严格测试 1: CPU 读写物理带宽测试 (判断 Cache 属性)\n";
    std::cout << "======================================================\n";
    
    std::vector<uint8_t> local_buffer(size, 0x55);
    
    // 测试 CPU 写入 (模拟 Producer 灌入相机数据)
    long long t0 = now_us();
    std::memcpy(ptr, local_buffer.data(), size);
    long long t1 = now_us();
    double write_ms = (t1 - t0) / 1000.0;
    double write_bw = (size / (1024.0 * 1024.0 * 1024.0)) / (write_ms / 1000.0);

    // 测试 CPU 读取 (模拟 VLM/YOLO 预处理时拉取数据)
    t0 = now_us();
    std::memcpy(local_buffer.data(), ptr, size);
    t1 = now_us();
    double read_ms = (t1 - t0) / 1000.0;
    double read_bw = (size / (1024.0 * 1024.0 * 1024.0)) / (read_ms / 1000.0);

    std::cout << "分配总大小: " << size / (1024.0 * 1024.0) << " MB\n";
    std::cout << "CPU 写入耗时: " << std::fixed << std::setprecision(2) << write_ms << " ms, 带宽: " << write_bw << " GB/s\n";
    std::cout << "CPU 读取耗时: " << read_ms << " ms, 带宽: " << read_bw << " GB/s\n\n";

    std::cout << "[硬件属性揭秘]:\n";
    std::cout << "-> 如果读取带宽低于 3 GB/s，说明 qcom,system 分配的是 Uncached (Write-Combining) 内存。\n";
    std::cout << "   (写入可能很快，但读取如同灾难，做零拷贝前务必注意下游是否用 CPU 读数据)\n";
    std::cout << "-> 如果读取带宽在 10 GB/s 以上，说明是 Cached 内存。\n";
}

void test_pipeline_simulation(int fd, void* ptr) {
    std::cout << "\n======================================================\n";
    std::cout << " 严格测试 2: 模拟 15fps 单 FD 轮转访问耗时\n";
    std::cout << "======================================================\n";
    
    struct dma_buf_sync sync = {};
    std::cout << "模拟 Producer 连续写入 8 个 Slot 的生命周期 (涵盖 Sync 开销)...\n";
    
    for (int i = 0; i < SLOT_COUNT; ++i) {
        long long t_start = now_us();
        
        // 1. Sync Start
        sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
        ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
        
        // 2. 写入单一 Slot 数据
        size_t offset = i * SLOT_SIZE;
        std::memset(static_cast<uint8_t*>(ptr) + offset, i, SLOT_SIZE);
        
        // 3. Sync End
        sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
        ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
        
        long long t_end = now_us();
        std::cout << "Slot " << i << " 完整写入周期耗时: " 
                  << std::fixed << std::setprecision(2) << (t_end - t_start) / 1000.0 << " ms\n";
    }
}

int main() {
    int fd = alloc_dma_buf(TOTAL_SIZE);
    if (fd < 0) {
        std::cerr << "DMA-BUF 分配失败!\n";
        return -1;
    }

    void* ptr = mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "mmap 失败!\n";
        close(fd);
        return -1;
    }

    test_memory_bandwidth(ptr, TOTAL_SIZE);
    test_pipeline_simulation(fd, ptr);

    munmap(ptr, TOTAL_SIZE);
    close(fd);
    return 0;
}
