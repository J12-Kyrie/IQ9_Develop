#include "slot_pool.h"

#include <cerrno>
#include <cstring>
#include <sstream>

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace dmabuf_producer {

SlotPool::SlotPool()
    : base_ptr_(nullptr),
      slot_stride_(0),
      buffer_size_(0),
      num_slots_(0),
      initialized_(false) {}

SlotPool::~SlotPool() {
    Shutdown();
}

bool SlotPool::Init(const char* heap_path, size_t slot_stride, int num_slots,
                    std::string* out_error) {
    if (initialized_) {
        if (out_error != nullptr) {
            *out_error = "SlotPool already initialized";
        }
        return false;
    }

    if (num_slots <= 0 || num_slots > kMaxSlots) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "Invalid num_slots=" << num_slots
                << " (must be 1.." << kMaxSlots << ")";
            *out_error = oss.str();
        }
        return false;
    }

    // Open DMA heap device (try primary, then fallback)
    int heap_fd = open(heap_path, O_RDONLY | O_CLOEXEC);
    if (heap_fd < 0) {
        // Fallback to generic system heap
        heap_fd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
        if (heap_fd < 0) {
            if (out_error != nullptr) {
                std::ostringstream oss;
                oss << "Failed to open DMA heap at '" << heap_path
                    << "' and fallback '/dev/dma_heap/system': "
                    << strerror(errno);
                *out_error = oss.str();
            }
            return false;
        }
    }

    if (slot_stride == 0) {
        if (out_error != nullptr) {
            *out_error = "slot_stride must be > 0";
        }
        close(heap_fd);
        return false;
    }

    size_t buffer_size = slot_stride * static_cast<size_t>(num_slots);
    if (buffer_size / static_cast<size_t>(num_slots) != slot_stride) {
        if (out_error != nullptr) {
            *out_error = "slot buffer size overflow";
        }
        close(heap_fd);
        return false;
    }

    struct dma_heap_allocation_data alloc_data;
    memset(&alloc_data, 0, sizeof(alloc_data));
    alloc_data.len = buffer_size;
    alloc_data.fd_flags = O_CLOEXEC | O_RDWR;
    alloc_data.heap_flags = 0;

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "DMA_HEAP_IOCTL_ALLOC failed for shared buffer "
                << "(size=" << buffer_size << "): " << strerror(errno);
            *out_error = oss.str();
        }
        close(heap_fd);
        return false;
    }

    void* ptr = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, alloc_data.fd, 0);
    if (ptr == MAP_FAILED) {
        if (out_error != nullptr) {
            std::ostringstream oss;
            oss << "mmap failed for shared buffer "
                << "(size=" << buffer_size << "): " << strerror(errno);
            *out_error = oss.str();
        }
        close(alloc_data.fd);
        close(heap_fd);
        return false;
    }

    slots_.resize(static_cast<size_t>(num_slots));
    for (int i = 0; i < num_slots; ++i) {
        Slot& slot = slots_[static_cast<size_t>(i)];
        slot.offset = static_cast<size_t>(i) * slot_stride;
        slot.state = IDLE;
    }
    shared_fd_ = DmaBufFd(alloc_data.fd);
    base_ptr_ = ptr;
    slot_stride_ = slot_stride;
    buffer_size_ = buffer_size;
    num_slots_ = num_slots;

    close(heap_fd);
    initialized_ = true;
    return true;
}

void SlotPool::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (base_ptr_ != nullptr && base_ptr_ != MAP_FAILED && buffer_size_ > 0) {
        munmap(base_ptr_, buffer_size_);
    }
    base_ptr_ = nullptr;
    shared_fd_.reset();

    for (size_t i = 0; i < slots_.size(); ++i) {
        slots_[i].offset = 0;
        slots_[i].state = IDLE;
    }
    slots_.clear();
    slot_stride_ = 0;
    buffer_size_ = 0;
    num_slots_ = 0;
    initialized_ = false;
}

int SlotPool::AcquireSlot() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (int i = 0; i < num_slots_; ++i) {
        if (slots_[static_cast<size_t>(i)].state == IDLE) {
            slots_[static_cast<size_t>(i)].state = IN_USE;
            return i;
        }
    }
    return -1;
}

bool SlotPool::SubmitSlot(int slot_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (slot_index < 0 || slot_index >= num_slots_) {
        return false;
    }
    if (slots_[static_cast<size_t>(slot_index)].state != IN_USE) {
        return false;
    }
    slots_[static_cast<size_t>(slot_index)].state = ON_FLIGHT;
    return true;
}

bool SlotPool::ReleaseSlot(int slot_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (slot_index < 0 || slot_index >= num_slots_) {
        return false;
    }
    if (slots_[static_cast<size_t>(slot_index)].state != ON_FLIGHT) {
        return false;
    }
    slots_[static_cast<size_t>(slot_index)].state = IDLE;
    return true;
}

bool SlotPool::RevertSlot(int slot_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (slot_index < 0 || slot_index >= num_slots_) {
        return false;
    }
    if (slots_[static_cast<size_t>(slot_index)].state != IN_USE) {
        return false;
    }
    slots_[static_cast<size_t>(slot_index)].state = IDLE;
    return true;
}

void* SlotPool::GetSlotPtr(int slot_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (slot_index < 0 || slot_index >= num_slots_ || base_ptr_ == nullptr) {
        return nullptr;
    }
    const Slot& slot = slots_[static_cast<size_t>(slot_index)];
    if (slot.offset >= buffer_size_) {
        return nullptr;
    }
    return static_cast<void*>(static_cast<uint8_t*>(base_ptr_) + slot.offset);
}

int SlotPool::GetSlotFd(int slot_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (slot_index < 0 || slot_index >= num_slots_) {
        return -1;
    }
    return shared_fd_.get();
}

size_t SlotPool::GetSlotStride(int slot_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (slot_index < 0 || slot_index >= num_slots_) {
        return 0;
    }
    return slot_stride_;
}

size_t SlotPool::GetSlotOffset(int slot_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (slot_index < 0 || slot_index >= num_slots_) {
        return 0;
    }
    return slots_[static_cast<size_t>(slot_index)].offset;
}

size_t SlotPool::GetBufferSize() {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_size_;
}

}  // namespace dmabuf_producer
