#ifndef DMABUF_CONSUMER_TYPES_H
#define DMABUF_CONSUMER_TYPES_H

#include <cstdint>
#include <unistd.h>

enum PixelFormat : uint32_t {
    PIXEL_FMT_NV12  = 0,
    PIXEL_FMT_RGB24 = 1,
};

struct IQ9_Task_Packet {
    uint32_t magic;            // 0x49513941 ("IQ9A")
    uint32_t slot_index;
    uint32_t width;
    uint32_t height;
    uint32_t channels;         // 3 for RGB24
    uint32_t data_type;        // PixelFormat
    uint64_t timestamp_ns;
};

struct IQ9_Release_Packet {
    uint32_t magic;            // 0x49513942 ("IQ9B")
    uint32_t slot_index;
    int32_t  status;           // 0 = success
};

struct IQ9_Init_Packet {
    uint32_t magic;       // 0x49513943 ("IQ9C")
    uint32_t slot_count;
    uint32_t slot_stride; // bytes per slot (4KB aligned)
    uint32_t buffer_size; // total bytes in shared DMA-BUF
    uint32_t width;
    uint32_t height;
    uint32_t channels;    // 3 for RGB24
    uint32_t data_type;   // PixelFormat
};

static_assert(sizeof(IQ9_Task_Packet) == 32, "IQ9_Task_Packet must be 32 bytes");
static_assert(sizeof(IQ9_Release_Packet) == 12, "IQ9_Release_Packet must be 12 bytes");
static_assert(sizeof(IQ9_Init_Packet) == 32, "IQ9_Init_Packet must be 32 bytes");

static const uint32_t IQ9_TASK_MAGIC    = 0x49513941;
static const uint32_t IQ9_RELEASE_MAGIC = 0x49513942;
static const uint32_t IQ9_INIT_MAGIC    = 0x49513943;

struct DmaBufFd {
    int fd;

    explicit DmaBufFd(int f = -1) : fd(f) {}
    ~DmaBufFd() { reset(); }

    DmaBufFd(const DmaBufFd&) = delete;
    DmaBufFd& operator=(const DmaBufFd&) = delete;

    DmaBufFd(DmaBufFd&& o) : fd(o.fd) { o.fd = -1; }
    DmaBufFd& operator=(DmaBufFd&& o) {
        if (this != &o) {
            reset();
            fd = o.fd;
            o.fd = -1;
        }
        return *this;
    }

    void reset() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    int get() const { return fd; }

    int release() {
        int tmp = fd;
        fd = -1;
        return tmp;
    }

    explicit operator bool() const { return fd >= 0; }
};

#endif // DMABUF_CONSUMER_TYPES_H
