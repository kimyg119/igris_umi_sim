#ifndef __SPSCRINGBUFFER_D21B481B_5988_4CDE_9814_C2C136A792C5_H__
#define __SPSCRINGBUFFER_D21B481B_5988_4CDE_9814_C2C136A792C5_H__

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace robros::shm::buffer {
template <typename T, std::size_t Capacity> class SpscRingBuffer {
  public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power-of-two (e.g. 2, 4, 8, 16...).");

    SpscRingBuffer() : writer_cnt_(0), reader_cnt_(0) {}

    /**
     * @brief Write a new item into the ring buffer (no overflow check).
     *
     * Single producer only. Overwrites unread data if the buffer
     * "wraps around" beyond Capacity.
     */
    void push(const T &item) {
        std::size_t currentWrite               = writer_cnt_.load(std::memory_order_relaxed);
        buffer_[currentWrite & (Capacity - 1)] = item;

        // Bump writer index
        writer_cnt_.store(currentWrite + 1, std::memory_order_release);
    }

    /**
     * @brief Pull (pop) the *oldest* unread item in FIFO order.
     *
     * Single consumer only. Returns true if an item was read,
     * false if the buffer was empty.
     */
    bool pull(T &out) {
        std::size_t currentRead  = reader_cnt_.load(std::memory_order_relaxed);
        std::size_t currentWrite = writer_cnt_.load(std::memory_order_acquire);

        if (currentRead == currentWrite) {
            // No unread items
            return false;
        }

        // Read the next item in FIFO order
        out = buffer_[currentRead & (Capacity - 1)];

        // Advance consumer read index
        reader_cnt_.store(currentRead + 1, std::memory_order_release);
        return true;
    }

    /**
     * @brief Pull the *most recent* item, skipping any intermediate ones.
     *
     * This emulates the pattern of your original snippet, which ignores
     * all but the newest. Returns true if an item was read, false if empty.
     */
    bool pull_latest(T &out) {
        std::size_t currentRead  = reader_cnt_.load(std::memory_order_relaxed);
        std::size_t currentWrite = writer_cnt_.load(std::memory_order_acquire);

        if (currentRead == currentWrite) {
            // Empty
            return false;
        }

        // The newest item is at (writer_cnt_ - 1)
        std::size_t lastIndex = (currentWrite - 1) & (Capacity - 1);
        out                   = buffer_[lastIndex];

        // Skip everything up to writer_cnt_
        reader_cnt_.store(currentWrite, std::memory_order_release);
        return true;
    }

    bool view_latest(T &out, size_t &index) const {
        std::size_t currentWrite = writer_cnt_.load(std::memory_order_acquire);
        std::size_t lastIndex    = (currentWrite - 1) & (Capacity - 1);
        out                      = buffer_[lastIndex];
        index                    = lastIndex;
        return true;
    }

    std::size_t view_latest_cnt() const {
        std::size_t currentWrite = writer_cnt_.load(std::memory_order_acquire);
        std::size_t lastIndex    = (currentWrite - 1) & (Capacity - 1);
        return lastIndex;
    }

    std::atomic<std::size_t> writer_cnt_;
    std::atomic<std::size_t> reader_cnt_;

  private:
    std::array<T, Capacity> buffer_;

    // Indices for writer/reader
};
}  // namespace robros::shm::buffer

#endif /* __SPSCRINGBUFFER_D21B481B_5988_4CDE_9814_C2C136A792C5_H__ */
