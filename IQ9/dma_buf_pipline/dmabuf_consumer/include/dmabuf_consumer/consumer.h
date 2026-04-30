#ifndef DMABUF_CONSUMER_CONSUMER_H
#define DMABUF_CONSUMER_CONSUMER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <dmabuf_producer/types.h>

namespace dmabuf_consumer {

class IpcReceiver;

class DmaBufConsumer {
public:
    static const int kMaxSlots = 8;

    struct Frame {
        int fd;                  // informational, managed by library — do NOT close
        void* data;              // pre-mapped RGB24 pointer — do NOT munmap
        uint32_t width;
        uint32_t height;
        uint32_t channels;
        uint32_t data_size;      // width * height * channels
        uint64_t timestamp_ns;
        uint32_t slot_index;
        uint32_t slot_stride;
        uint64_t slot_offset;
    };

    DmaBufConsumer();
    ~DmaBufConsumer();

    DmaBufConsumer(const DmaBufConsumer&) = delete;
    DmaBufConsumer& operator=(const DmaBufConsumer&) = delete;

    bool Init(const char* socket_path, std::string* out_error);
    void Shutdown();

    // Blocking. Returns false with empty error on EOF (producer disconnected).
    bool RecvFrame(Frame* out, std::string* out_error);

    // Map an already-received packet into a Frame (validates magic, bounds,
    // performs DMA_BUF_SYNC_START|READ). Useful when the packet was received
    // externally (e.g. via a relay UDS) rather than through RecvFrame.
    bool MapFrame(const IQ9_Task_Packet& pkt, Frame* out, std::string* out_error);

    // SYNC_END -> send release. No munmap/close (deferred to Shutdown).
    bool ReleaseFrame(const Frame& frame, std::string* out_error);

private:
    IpcReceiver* receiver_;
    bool initialized_;
    int shared_fd_;
    void* base_ptr_;
    size_t slot_stride_;
    size_t buffer_size_;
    uint64_t sync_read_start_total_us_;
    uint64_t sync_read_end_total_us_;
    uint64_t sync_read_start_max_us_;
    uint64_t sync_read_end_max_us_;
    uint64_t sync_read_samples_;
    int slot_count_;
};

}  // namespace dmabuf_consumer

#endif  // DMABUF_CONSUMER_CONSUMER_H
