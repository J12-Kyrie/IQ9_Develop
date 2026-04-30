#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <cstring>
#include <sys/wait.h>

// 配置：1920*1080*4 (RGBA) * 8 Slots
constexpr size_t SLOT_SIZE = 1920 * 1080 * 4; 
constexpr int SLOT_COUNT = 8;
constexpr size_t TOTAL_SIZE = SLOT_SIZE * SLOT_COUNT;
constexpr const char* HEAP_PATH = "/dev/dma_heap/qcom,system";

// IPC 辅助函数：通过 Socket 发送 FD
int send_fd(int sock, int fd) {
    struct msghdr msg = {0};
    char buf[CMSG_SPACE(sizeof(int))];
    memset(buf, 0, sizeof(buf));

    struct iovec io = { .iov_base = (void*)"FD", .iov_len = 2 };
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    *((int *)CMSG_DATA(cmsg)) = fd;

    return sendmsg(sock, &msg, 0);
}

// IPC 辅助函数：通过 Socket 接收 FD
int recv_fd(int sock) {
    struct msghdr msg = {0};
    char buf[CMSG_SPACE(sizeof(int))];
    struct iovec io = { .iov_base = buf, .iov_len = sizeof(buf) };
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    if (recvmsg(sock, &msg, 0) < 0) return -1;
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    return *((int *)CMSG_DATA(cmsg));
}

inline long long now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ======================================================
// Consumer 进程逻辑 (模拟推理服务)
// ======================================================
void run_consumer(int sock) {
    int fd = recv_fd(sock);
    if (fd < 0) return;

    void* ptr = mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    
    // 模拟等待 Producer 准备好数据
    char sync_byte;
    read(sock, &sync_byte, 1);

    std::cout << "\n[Consumer] 检测到数据已到达，准备执行 SYNC_START (Invalidate Cache)..." << std::endl;
    
    struct dma_buf_sync sync = { .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ };
    long long t0 = now_us();
    ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
    long long t1 = now_us();

    // 模拟读取数据（确保数据被拉入 CPU Cache）
    volatile uint8_t sum = 0;
    uint8_t* u8ptr = static_cast<uint8_t*>(ptr);
    for(size_t i=0; i<TOTAL_SIZE; i += 4096) sum += u8ptr[i];

    std::cout << "[Consumer] 冲刷整个 DMA_BUF (63.3MB) 耗时: " 
              << (t1 - t0) / 1000.0 << " ms" << std::endl;

    munmap(ptr, TOTAL_SIZE);
    close(fd);
}

// ======================================================
// Producer 进程逻辑 (模拟相机/感知服务)
// ======================================================
void run_producer(int sock) {
    int heap_fd = open(HEAP_PATH, O_RDWR);
    struct dma_heap_allocation_data alloc_data = { .len = TOTAL_SIZE, .fd_flags = O_RDWR };
    ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data);
    int fd = alloc_data.fd;

    void* ptr = mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    
    // 关键：必须先脏化数据 (Dirtying the cache)
    std::memset(ptr, 0xEF, TOTAL_SIZE);

    // 发送 FD 给 Consumer
    send_fd(sock, fd);

    std::cout << "[Producer] 准备执行 SYNC_END (Flush Cache to RAM)..." << std::endl;

    struct dma_buf_sync sync = { .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE };
    long long t0 = now_us();
    ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
    long long t1 = now_us();

    std::cout << "[Producer] 冲刷整个 DMA_BUF (63.3MB) 耗时: " 
              << (t1 - t0) / 1000.0 << " ms" << std::endl;

    // 通知 Consumer 可以开始读了
    write(sock, "G", 1);

    munmap(ptr, TOTAL_SIZE);
    close(fd);
    close(heap_fd);
}

int main() {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) < 0) return 1;

    pid_t pid = fork();
    if (pid == 0) { // Child
        close(sv[0]);
        run_consumer(sv[1]);
    } else { // Parent
        close(sv[1]);
        run_producer(sv[0]);
        wait(NULL);
    }
    return 0;
}
