#include "dmabuf_producer/producer.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sstream>

#include <linux/dma-buf.h>
#include <sys/ioctl.h>

#include "ipc_sender.h"
#include "nv12_to_rgb.h"
#include "slot_pool.h"

namespace dmabuf_producer {

// 4KB alignment macro
static inline size_t AlignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

DmaBufProducer::DmaBufProducer()
    : slot_pool_(nullptr), ipc_sender_(nullptr), frame_counter_(0),
      width_(0), height_(0), slot_count_(0), slot_stride_(0),
      buffer_size_(0),
      sync_write_start_total_us_(0),
      sync_write_end_total_us_(0),
      sync_write_start_max_us_(0),
      sync_write_end_max_us_(0),
      sync_write_samples_(0),
      relay_mode_(false),
      direct_released_(0) {}

DmaBufProducer::~DmaBufProducer() {
    Shutdown();
}

bool DmaBufProducer::Init(const Config& cfg, std::string* out_error) {
    const char* socket_path = cfg.socket_path;
    if (socket_path == nullptr) {
        socket_path = "/tmp/dmabuf_ipc.sock";
    }

    const char* heap_path = cfg.heap_path;
    if (heap_path == nullptr) {
        heap_path = "/dev/dma_heap/qcom,system";
    }

    int slot_count = cfg.slot_count;
    if (slot_count <= 0) {
        slot_count = 4;
    }

    if (cfg.width <= 0 || cfg.height <= 0) {
        if (out_error != nullptr) {
            *out_error = "DmaBufProducer::Init: width and height must be > 0";
        }
        return false;
    }

    // Compute slot stride: RGB24 = width * height * 3, 4KB aligned
    size_t raw_size = static_cast<size_t>(cfg.width) *
                      static_cast<size_t>(cfg.height) * 3;
    size_t slot_stride = AlignUp(raw_size, 4096);
    size_t buffer_size = slot_stride * static_cast<size_t>(slot_count);

    // Init slot pool
    slot_pool_ = new SlotPool();
    if (!slot_pool_->Init(heap_path, slot_stride, slot_count, out_error)) {
        delete slot_pool_;
        slot_pool_ = nullptr;
        return false;
    }

    // Init IPC sender (bind + listen)
    ipc_sender_ = new IpcSender();
    if (!ipc_sender_->Init(socket_path, out_error)) {
        slot_pool_->Shutdown();
        delete slot_pool_;
        slot_pool_ = nullptr;
        delete ipc_sender_;
        ipc_sender_ = nullptr;
        return false;
    }

    // Block until consumer connects
    if (!ipc_sender_->AcceptClient(out_error)) {
        ipc_sender_->Shutdown();
        slot_pool_->Shutdown();
        delete slot_pool_;
        slot_pool_ = nullptr;
        delete ipc_sender_;
        ipc_sender_ = nullptr;
        return false;
    }

    // Store config for handshake and future use
    width_ = cfg.width;
    height_ = cfg.height;
    slot_count_ = slot_count;
    slot_stride_ = slot_stride;
    buffer_size_ = buffer_size;
    relay_mode_ = cfg.relay_mode;
    direct_released_.store(0, std::memory_order_relaxed);
    sync_write_start_total_us_ = 0;
    sync_write_end_total_us_ = 0;
    sync_write_start_max_us_ = 0;
    sync_write_end_max_us_ = 0;
    sync_write_samples_ = 0;

    // Build init packet for handshake
    IQ9_Init_Packet init_pkt;
    memset(&init_pkt, 0, sizeof(init_pkt));
    init_pkt.magic = IQ9_INIT_MAGIC;
    init_pkt.slot_count = static_cast<uint32_t>(slot_count);
    init_pkt.slot_stride = static_cast<uint32_t>(slot_stride);
    init_pkt.buffer_size = static_cast<uint32_t>(buffer_size);
    init_pkt.width = static_cast<uint32_t>(cfg.width);
    init_pkt.height = static_cast<uint32_t>(cfg.height);
    init_pkt.channels = 3;
    init_pkt.data_type = PIXEL_FMT_RGB24;

    // Send handshake: init packet + one shared slot fd via SCM_RIGHTS
    int shared_fd = slot_pool_->GetSlotFd(0);
    if (!ipc_sender_->SendInit(init_pkt, shared_fd, out_error)) {
        ipc_sender_->Shutdown();
        slot_pool_->Shutdown();
        delete slot_pool_;
        slot_pool_ = nullptr;
        delete ipc_sender_;
        ipc_sender_ = nullptr;
        return false;
    }

    frame_counter_ = 0;
    return true;
}

void DmaBufProducer::Shutdown() {
    if (ipc_sender_ != nullptr || slot_pool_ != nullptr) {
        std::fprintf(stderr, "[producer] direct_released=%d\n",
                     direct_released_.load(std::memory_order_relaxed));
        if (sync_write_samples_ > 0) {
            const uint64_t avg_start = sync_write_start_total_us_ / sync_write_samples_;
            const uint64_t avg_end = sync_write_end_total_us_ / sync_write_samples_;
            std::fprintf(stderr,
                         "[producer] ioctl_sync_write_us samples=%lu start_avg=%lu start_max=%lu end_avg=%lu end_max=%lu\n",
                         static_cast<unsigned long>(sync_write_samples_),
                         static_cast<unsigned long>(avg_start),
                         static_cast<unsigned long>(sync_write_start_max_us_),
                         static_cast<unsigned long>(avg_end),
                         static_cast<unsigned long>(sync_write_end_max_us_));
        }
    }
    if (ipc_sender_ != nullptr) {
        ipc_sender_->Shutdown();
        delete ipc_sender_;
        ipc_sender_ = nullptr;
    }
    if (slot_pool_ != nullptr) {
        slot_pool_->Shutdown();
        delete slot_pool_;
        slot_pool_ = nullptr;
    }
}

bool DmaBufProducer::SubmitFrame(const uint8_t* nv12_data, int width,
                                 int height, int y_stride, int uv_offset,
                                 uint64_t timestamp_ns,
                                 std::string* out_error,
                                 uint32_t* out_slot_index) {
    if (slot_pool_ == nullptr || ipc_sender_ == nullptr) {
        if (out_error != nullptr) {
            *out_error = "DmaBufProducer not initialized";
        }
        return false;
    }

    // Try to reclaim released slots first
    PollReleases();

    // Acquire an idle slot
    int slot_idx = slot_pool_->AcquireSlot();
    if (slot_idx < 0) {
        if (out_error != nullptr) {
            *out_error = "no free slots";
        }
        return false;
    }

    int dma_fd = slot_pool_->GetSlotFd(slot_idx);
    uint8_t* slot_ptr = static_cast<uint8_t*>(slot_pool_->GetSlotPtr(slot_idx));

    if (dma_fd < 0 || slot_ptr == nullptr) {
        slot_pool_->RevertSlot(slot_idx);
        if (out_error != nullptr) {
            *out_error = "invalid slot state";
        }
        return false;
    }

    // DMA_BUF_IOCTL_SYNC: begin CPU write access (cache invalidate)
    struct dma_buf_sync sync_start;
    memset(&sync_start, 0, sizeof(sync_start));
    sync_start.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE;
    auto sync_start_t0 = std::chrono::steady_clock::now();
    int sync_start_rc = ioctl(dma_fd, DMA_BUF_IOCTL_SYNC, &sync_start);
    uint64_t sync_start_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - sync_start_t0).count());
    sync_write_start_total_us_ += sync_start_us;
    if (sync_start_us > sync_write_start_max_us_) {
        sync_write_start_max_us_ = sync_start_us;
    }
    if (sync_start_rc < 0) {
        slot_pool_->RevertSlot(slot_idx);
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "DMA_BUF_IOCTL_SYNC(SYNC_START|SYNC_WRITE) failed: "
                << strerror(errno);
            *out_error = oss.str();
        }
        return false;
    }

    // NV12 -> RGB24 conversion into CMA slot
    Nv12ToRgb24(nv12_data, uv_offset, slot_ptr, width, height, y_stride);

    // DMA_BUF_IOCTL_SYNC: end CPU write access before sending
    struct dma_buf_sync sync_end;
    memset(&sync_end, 0, sizeof(sync_end));
    sync_end.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    auto sync_end_t0 = std::chrono::steady_clock::now();
    int sync_end_rc = ioctl(dma_fd, DMA_BUF_IOCTL_SYNC, &sync_end);
    uint64_t sync_end_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - sync_end_t0).count());
    sync_write_end_total_us_ += sync_end_us;
    if (sync_end_us > sync_write_end_max_us_) {
        sync_write_end_max_us_ = sync_end_us;
    }
    if (sync_end_rc < 0) {
        slot_pool_->RevertSlot(slot_idx);
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "DMA_BUF_IOCTL_SYNC(SYNC_END|SYNC_WRITE) failed: "
                << strerror(errno);
            *out_error = oss.str();
        }
        return false;
    }
    ++sync_write_samples_;

    // Build task packet
    IQ9_Task_Packet pkt;
    memset(&pkt, 0, sizeof(pkt));
    pkt.magic = IQ9_TASK_MAGIC;
    pkt.slot_index = static_cast<uint32_t>(slot_idx);
    pkt.width = static_cast<uint32_t>(width);
    pkt.height = static_cast<uint32_t>(height);
    pkt.channels = 3;
    pkt.data_type = PIXEL_FMT_RGB24;
    pkt.timestamp_ns = timestamp_ns;

    // Plain send — no SCM_RIGHTS, no fd, consumer uses pre-mapped slot
    if (!relay_mode_) {
        if (!ipc_sender_->SendFrame(pkt, out_error)) {
            slot_pool_->RevertSlot(slot_idx);
            return false;
        }
    }

    // Mark slot as in-flight
    if (!slot_pool_->SubmitSlot(slot_idx)) {
        slot_pool_->RevertSlot(slot_idx);
        if (out_error) *out_error = "SubmitSlot failed: unexpected slot state";
        return false;
    }

    if (out_slot_index) *out_slot_index = static_cast<uint32_t>(slot_idx);

    ++frame_counter_;
    return true;
}

int DmaBufProducer::PollReleases() {
    std::lock_guard<std::mutex> lock(poll_mutex_);

    if (ipc_sender_ == nullptr || slot_pool_ == nullptr) {
        return 0;
    }

    int count = 0;
    IQ9_Release_Packet release_pkt;

    while (ipc_sender_->PollRelease(&release_pkt)) {
        if (slot_pool_->ReleaseSlot(static_cast<int>(release_pkt.slot_index))) {
            ++count;
        }
    }

    return count;
}

bool DmaBufProducer::ReleaseSlotDirect(uint32_t slot_index) {
    if (slot_pool_ == nullptr) return false;
    const bool released = slot_pool_->ReleaseSlot(static_cast<int>(slot_index));
    if (released) {
        direct_released_.fetch_add(1, std::memory_order_relaxed);
    }
    return released;
}

}  // namespace dmabuf_producer
