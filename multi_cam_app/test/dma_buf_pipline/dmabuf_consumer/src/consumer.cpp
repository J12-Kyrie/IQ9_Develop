#include <dmabuf_consumer/consumer.h>
#include "ipc_receiver.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/dma-buf.h>

namespace dmabuf_consumer {

DmaBufConsumer::DmaBufConsumer()
    : receiver_(new IpcReceiver()),
      initialized_(false),
      shared_fd_(-1),
      base_ptr_(nullptr),
      slot_stride_(0),
      buffer_size_(0),
      sync_read_start_total_us_(0),
      sync_read_end_total_us_(0),
      sync_read_start_max_us_(0),
      sync_read_end_max_us_(0),
      sync_read_samples_(0),
      slot_count_(0) {}

DmaBufConsumer::~DmaBufConsumer() {
    Shutdown();
    delete receiver_;
}

bool DmaBufConsumer::Init(const char* socket_path, std::string* out_error) {
    if (initialized_) {
        if (out_error != nullptr) {
            *out_error = "consumer already initialized";
        }
        return false;
    }

    if (!receiver_->Connect(socket_path, out_error)) {
        return false;
    }

    // Receive handshake: init packet + shared fd
    IQ9_Init_Packet init_pkt;
    std::memset(&init_pkt, 0, sizeof(init_pkt));

    int shared_fd = -1;

    if (!receiver_->RecvInit(&init_pkt, &shared_fd, out_error)) {
        receiver_->Shutdown();
        return false;
    }

    int slot_count = static_cast<int>(init_pkt.slot_count);
    if (slot_count <= 0 || slot_count > kMaxSlots) {
        if (out_error != nullptr) {
            *out_error = "invalid slot_count in init packet";
        }
        if (shared_fd >= 0) ::close(shared_fd);
        receiver_->Shutdown();
        return false;
    }

    static const size_t kMaxBufferSize = 128 * 1024 * 1024;  // 128 MB
    size_t slot_stride = static_cast<size_t>(init_pkt.slot_stride);
    size_t buffer_size = static_cast<size_t>(init_pkt.buffer_size);
    if (slot_stride == 0 || buffer_size == 0 || buffer_size > kMaxBufferSize) {
        if (out_error != nullptr) {
            *out_error = "slot_stride/buffer_size out of range";
        }
        if (shared_fd >= 0) ::close(shared_fd);
        receiver_->Shutdown();
        return false;
    }
    if (slot_stride * static_cast<size_t>(slot_count) != buffer_size) {
        if (out_error != nullptr) {
            *out_error = "buffer_size mismatch with slot_count*slot_stride";
        }
        if (shared_fd >= 0) ::close(shared_fd);
        receiver_->Shutdown();
        return false;
    }

    void* ptr = ::mmap(NULL, buffer_size, PROT_READ, MAP_SHARED, shared_fd, 0);
    if (ptr == MAP_FAILED) {
        if (out_error != nullptr) {
            *out_error = std::string("mmap shared buffer failed: ") + std::strerror(errno);
        }
        ::close(shared_fd);
        receiver_->Shutdown();
        return false;
    }

    shared_fd_ = shared_fd;
    base_ptr_ = ptr;
    slot_stride_ = slot_stride;
    buffer_size_ = buffer_size;
    sync_read_start_total_us_ = 0;
    sync_read_end_total_us_ = 0;
    sync_read_start_max_us_ = 0;
    sync_read_end_max_us_ = 0;
    sync_read_samples_ = 0;
    slot_count_ = slot_count;
    initialized_ = true;
    return true;
}

void DmaBufConsumer::Shutdown() {
    if (initialized_) {
        if (sync_read_samples_ > 0) {
            const uint64_t avg_start = sync_read_start_total_us_ / sync_read_samples_;
            const uint64_t avg_end = sync_read_end_total_us_ / sync_read_samples_;
            std::fprintf(stderr,
                         "[consumer] ioctl_sync_read_us samples=%lu start_avg=%lu start_max=%lu end_avg=%lu end_max=%lu\n",
                         static_cast<unsigned long>(sync_read_samples_),
                         static_cast<unsigned long>(avg_start),
                         static_cast<unsigned long>(sync_read_start_max_us_),
                         static_cast<unsigned long>(avg_end),
                         static_cast<unsigned long>(sync_read_end_max_us_));
        }
        if (base_ptr_ != nullptr && base_ptr_ != MAP_FAILED && buffer_size_ > 0) {
            ::munmap(base_ptr_, buffer_size_);
        }
        if (shared_fd_ >= 0) {
            ::close(shared_fd_);
        }
        shared_fd_ = -1;
        base_ptr_ = nullptr;
        slot_stride_ = 0;
        buffer_size_ = 0;
        slot_count_ = 0;
        receiver_->Shutdown();
        initialized_ = false;
    }
}

bool DmaBufConsumer::MapFrame(const IQ9_Task_Packet& pkt, Frame* out,
                              std::string* out_error) {
    if (!initialized_) {
        if (out_error != nullptr) {
            *out_error = "consumer not initialized";
        }
        return false;
    }

    if (pkt.magic != IQ9_TASK_MAGIC) {
        if (out_error != nullptr) {
            *out_error = "invalid task packet magic";
        }
        return false;
    }

    int idx = static_cast<int>(pkt.slot_index);
    if (idx < 0 || idx >= slot_count_) {
        if (out_error != nullptr) {
            *out_error = "slot_index " + std::to_string(idx) +
                         " out of range (slot_count=" +
                         std::to_string(slot_count_) + ")";
        }
        return false;
    }

    // Validate frame dimensions fit within the shared buffer window
    uint64_t raw_size = static_cast<uint64_t>(pkt.width) *
                        static_cast<uint64_t>(pkt.height) *
                        static_cast<uint64_t>(pkt.channels);
    uint64_t offset = static_cast<uint64_t>(idx) * static_cast<uint64_t>(slot_stride_);
    if (raw_size == 0 ||
        offset > static_cast<uint64_t>(buffer_size_) ||
        raw_size > static_cast<uint64_t>(slot_stride_) ||
        offset + raw_size > static_cast<uint64_t>(buffer_size_)) {
        if (out_error != nullptr) {
            *out_error = "frame dimensions exceed slot window/buffer size";
        }
        return false;
    }
    uint32_t data_size = static_cast<uint32_t>(raw_size);

    // DMA_BUF_SYNC_START|SYNC_READ for cache coherency on pre-mapped buffer
    struct dma_buf_sync sync_start;
    std::memset(&sync_start, 0, sizeof(sync_start));
    sync_start.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    auto sync_start_t0 = std::chrono::steady_clock::now();
    int sync_start_rc = ::ioctl(shared_fd_, DMA_BUF_IOCTL_SYNC, &sync_start);
    uint64_t sync_start_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - sync_start_t0).count());
    sync_read_start_total_us_ += sync_start_us;
    if (sync_start_us > sync_read_start_max_us_) {
        sync_read_start_max_us_ = sync_start_us;
    }
    if (sync_start_rc < 0) {
        // Non-fatal: some heaps don't require explicit sync
    }

    out->fd = shared_fd_;
    out->data = static_cast<void*>(static_cast<uint8_t*>(base_ptr_) + offset);
    out->width = pkt.width;
    out->height = pkt.height;
    out->channels = pkt.channels;
    out->data_size = data_size;
    out->timestamp_ns = pkt.timestamp_ns;
    out->slot_index = pkt.slot_index;
    out->slot_stride = static_cast<uint32_t>(slot_stride_);
    out->slot_offset = offset;

    return true;
}

bool DmaBufConsumer::RecvFrame(Frame* out, std::string* out_error) {
    if (!initialized_) {
        if (out_error != nullptr) {
            *out_error = "consumer not initialized";
        }
        return false;
    }

    IQ9_Task_Packet pkt;
    std::memset(&pkt, 0, sizeof(pkt));

    if (!receiver_->RecvFrame(&pkt, out_error)) {
        return false;
    }

    return MapFrame(pkt, out, out_error);
}

bool DmaBufConsumer::ReleaseFrame(const Frame& frame, std::string* out_error) {
    // SYNC_END|SYNC_READ on the pre-mapped slot, then send release
    int idx = static_cast<int>(frame.slot_index);
    if (idx >= 0 && idx < slot_count_ && shared_fd_ >= 0) {
        struct dma_buf_sync sync_end;
        std::memset(&sync_end, 0, sizeof(sync_end));
        sync_end.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
        auto sync_end_t0 = std::chrono::steady_clock::now();
        int sync_end_rc = ::ioctl(shared_fd_, DMA_BUF_IOCTL_SYNC, &sync_end);
        uint64_t sync_end_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - sync_end_t0).count());
        sync_read_end_total_us_ += sync_end_us;
        if (sync_end_us > sync_read_end_max_us_) {
            sync_read_end_max_us_ = sync_end_us;
        }
        if (sync_end_rc < 0) {
            // Non-fatal for some heaps
        }
        ++sync_read_samples_;
    }

    IQ9_Release_Packet pkt;
    pkt.magic = IQ9_RELEASE_MAGIC;
    pkt.slot_index = frame.slot_index;
    pkt.status = 0;

    return receiver_->SendRelease(pkt, out_error);
}

}  // namespace dmabuf_consumer
