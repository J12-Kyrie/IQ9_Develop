#ifndef DMABUF_PRODUCER_SLOT_POOL_H
#define DMABUF_PRODUCER_SLOT_POOL_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "dmabuf_producer/types.h"

namespace dmabuf_producer {

class SlotPool {
public:
    static const int kMaxSlots = 8;

    enum SlotState {
        IDLE = 0,
        IN_USE = 1,
        ON_FLIGHT = 2,
    };

    SlotPool();
    ~SlotPool();

    SlotPool(const SlotPool&) = delete;
    SlotPool& operator=(const SlotPool&) = delete;

    bool Init(const char* heap_path, size_t slot_stride, int num_slots,
              std::string* out_error);
    void Shutdown();

    // Returns slot index (IDLE->IN_USE), -1 if none free
    int AcquireSlot();

    // IN_USE -> ON_FLIGHT
    bool SubmitSlot(int slot_index);

    // ON_FLIGHT -> IDLE
    bool ReleaseSlot(int slot_index);

    // IN_USE -> IDLE (rollback on error before SubmitSlot)
    bool RevertSlot(int slot_index);

    void* GetSlotPtr(int slot_index);
    int GetSlotFd(int slot_index);
    size_t GetSlotStride(int slot_index);
    size_t GetSlotOffset(int slot_index);
    size_t GetBufferSize();

private:
    struct Slot {
        size_t offset;
        SlotState state;

        Slot() : offset(0), state(IDLE) {}
    };

    std::vector<Slot> slots_;
    DmaBufFd shared_fd_;
    void* base_ptr_;
    size_t slot_stride_;
    size_t buffer_size_;
    int num_slots_;
    std::mutex mutex_;
    bool initialized_;
};

}  // namespace dmabuf_producer

#endif  // DMABUF_PRODUCER_SLOT_POOL_H
