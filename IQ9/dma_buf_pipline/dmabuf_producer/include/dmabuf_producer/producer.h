#ifndef DMABUF_PRODUCER_PRODUCER_H
#define DMABUF_PRODUCER_PRODUCER_H

#include <cstdint>
#include <atomic>
#include <mutex>
#include <string>

namespace dmabuf_producer {

class SlotPool;
class IpcSender;

class DmaBufProducer {
public:
    struct Config {
        const char* socket_path;  // default "/tmp/dmabuf_ipc.sock"
        const char* heap_path;    // default "/dev/dma_heap/qcom,system"
        int slot_count;           // default 4
        int width;                // frame width (e.g. 1920)
        int height;               // frame height (e.g. 1080)
        bool relay_mode = false;  // skip IPC send; caller owns slot release
    };

    DmaBufProducer();
    ~DmaBufProducer();

    // Non-copyable, non-movable (pimpl pointers)
    DmaBufProducer(const DmaBufProducer&) = delete;
    DmaBufProducer& operator=(const DmaBufProducer&) = delete;

    bool Init(const Config& cfg, std::string* out_error);
    void Shutdown();

    // NV12 pixel data -> RGB24 CMA slot -> send fd to consumer
    // uv_offset: from GstVideoMeta::offset[1], NOT y_stride*height
    // out_slot_index: if non-null, receives the slot index used (relay_mode only)
    bool SubmitFrame(const uint8_t* nv12_data, int width, int height,
                     int y_stride, int uv_offset, uint64_t timestamp_ns,
                     std::string* out_error, uint32_t* out_slot_index = nullptr);

    // Non-blocking, returns number of slots reclaimed
    int PollReleases();

    // Directly release a slot by index (used in relay_mode by the caller)
    bool ReleaseSlotDirect(uint32_t slot_index);

private:
    SlotPool* slot_pool_;
    IpcSender* ipc_sender_;
    int frame_counter_;
    int width_;
    int height_;
    int slot_count_;
    size_t slot_stride_;
    size_t buffer_size_;
    uint64_t sync_write_start_total_us_;
    uint64_t sync_write_end_total_us_;
    uint64_t sync_write_start_max_us_;
    uint64_t sync_write_end_max_us_;
    uint64_t sync_write_samples_;
    bool relay_mode_;
    std::atomic<int> direct_released_;
    std::mutex poll_mutex_;
};

}  // namespace dmabuf_producer

#endif  // DMABUF_PRODUCER_PRODUCER_H
