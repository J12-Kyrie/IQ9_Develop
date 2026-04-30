#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <linux/dma-buf.h>
#include <cstring>
#include <chrono>

constexpr size_t TOTAL_SIZE = 1920 * 1080 * 4 * 8;
const char* SOCKET_PATH = "/tmp/iq9_cache_test.sock";

int recv_fd(int sock) {
    struct msghdr msg = {0};
    char buf[CMSG_SPACE(sizeof(int))];
    struct iovec io = { .iov_base = buf, .iov_len = sizeof(buf) };
    msg.msg_iov = &io; msg.msg_iovlen = 1;
    msg.msg_control = buf; msg.msg_controllen = sizeof(buf);
    if (recvmsg(sock, &msg, 0) < 0) return -1;
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    return *((int *)CMSG_DATA(cmsg));
}

int main() {
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = { .sun_family = AF_UNIX };
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path)-1);
    
    while(connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        usleep(100000);
    }

    int dma_fd = recv_fd(sock);
    void* ptr = mmap(nullptr, TOTAL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, dma_fd, 0);

    // 【核心测试】测量 SYNC_START (Invalidate) 耗时
    struct dma_buf_sync sync = { .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ };
    auto t0 = std::chrono::steady_clock::now();
    ioctl(dma_fd, DMA_BUF_IOCTL_SYNC, &sync);
    auto t1 = std::chrono::steady_clock::now();

    // 模拟真实读取（强迫 Cache Line 填充）
    volatile uint8_t* u8ptr = static_cast<uint8_t*>(ptr);
    uint32_t sum = 0;
    for(size_t i=0; i<TOTAL_SIZE; i+=4096) sum += u8ptr[i];

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "[Consumer] Invalidate 63.3MB 缓存耗时: " << duration / 1000.0 << " ms" << std::endl;

    write(sock, "D", 1);
    close(dma_fd); close(sock);
    return 0;
}
