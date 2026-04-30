#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <cstring>
#include <chrono>

constexpr size_t SLOT_SIZE = 1920 * 1080 * 4;
constexpr int SLOT_COUNT = 8;
constexpr size_t TOTAL_SIZE = SLOT_SIZE * SLOT_COUNT;
const char* SOCKET_PATH = "/tmp/iq9_cache_test.sock";

// 发送 FD 的辅助函数
void send_fd(int sock, int fd) {
    struct msghdr msg = {0};
    char buf[CMSG_SPACE(sizeof(int))];
    struct iovec io = { .iov_base = (void*)"FD", .iov_len = 2 };
    msg.msg_iov = &io; msg.msg_iovlen = 1;
    msg.msg_control = buf; msg.msg_controllen = sizeof(buf);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET; cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    *((int *)CMSG_DATA(cmsg)) = fd;
    sendmsg(sock, &msg, 0);
}

int main() {
    // 1. 分配 DMA-BUF
    int heap_fd = open("/dev/dma_heap/qcom,system", O_RDWR);
    struct dma_heap_allocation_data alloc = { .len = TOTAL_SIZE, .fd_flags = O_RDWR };
    ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc);
    int dma_fd = alloc.fd;

    // 2. 映射并“弄脏” Cache
    void* ptr = mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, dma_fd, 0);
    std::memset(ptr, 0x42, TOTAL_SIZE); // 确保数据在 Cache 中是 Dirty 状态

    // 3. 建立 Socket 等待 Consumer 连接
    unlink(SOCKET_PATH);
    int serv_sock = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = { .sun_family = AF_UNIX };
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path)-1);
    bind(serv_sock, (struct sockaddr*)&addr, sizeof(addr));
    listen(serv_sock, 1);
    
    std::cout << "[Producer] 等待 Consumer 连接..." << std::endl;
    int conn_sock = accept(serv_sock, nullptr, nullptr);
    
    // 4. 发送 FD 
    send_fd(conn_sock, dma_fd);

    // 5. 【核心测试】测量 SYNC_END (Flush) 耗时
    struct dma_buf_sync sync = { .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE };
    auto t0 = std::chrono::steady_clock::now();
    if (ioctl(dma_fd, DMA_BUF_IOCTL_SYNC, &sync) < 0) perror("ioctl");
    auto t1 = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "[Producer] 冲刷 63.3MB 缓存耗时: " << duration / 1000.0 << " ms" << std::endl;

    // 保持进程直到 Consumer 完成
    char sync_end;
    read(conn_sock, &sync_end, 1);
    
    close(dma_fd); close(heap_fd); unlink(SOCKET_PATH);
    return 0;
}
