#ifndef __SPMC_BROADCAST_DROP_RINGBUFFER_H__
#define __SPMC_BROADCAST_DROP_RINGBUFFER_H__

#include "igris_c_shm/logger/logger.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

#ifdef __linux__
#include <cerrno>
#include <climits>
#include <csignal>
#include <ctime>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifndef CONSUMER_ID_MAX_LEN
#define CONSUMER_ID_MAX_LEN 64
#endif

#ifndef CACHELINE_SIZE
#define CACHELINE_SIZE 64
#endif

namespace robros::shm::buffer {

#ifdef __linux__
static inline int futex_wait_u32(const uint32_t *addr, uint32_t expected) {
    return syscall(SYS_futex, addr, FUTEX_WAIT, expected, nullptr, nullptr, 0);
}

static inline int futex_wait_u32_timed(const uint32_t *addr, uint32_t expected, const timespec *timeout) {
    // timeout is relative for FUTEX_WAIT
    return syscall(SYS_futex, addr, FUTEX_WAIT, expected, timeout, nullptr, 0);
}

static inline int futex_wake_u32(const uint32_t *addr, int count) {
    return syscall(SYS_futex, addr, FUTEX_WAKE, count, nullptr, nullptr, 0);
}
#endif

/**
 * SPMC (Single Producer / Multi Consumer) broadcast ring buffer.
 *
 * - Single producer
 * - Multiple consumers
 * - All consumers observe the same logical stream
 * - Slow consumers MAY miss data (data loss is allowed)
 * - Producer push() NEVER blocks and NEVER fails
 *
 * Safety guarantees:
 * - Consumers will never read partially-written (torn) data
 * - Slot-level sequence numbers (seq) are used to detect overwrite
 *
 * Optional notification (Linux futex):
 * - push() can wake sleeping consumers
 * - consumers can wait on notify_seq_raw_ and then pull() in a loop
 *
 * NOTE:
 * - futex works on a 32-bit word. We store notify_seq_raw_ as a raw uint32_t
 *   in shared memory and access it via std::atomic_ref<uint32_t>.
 */
template <typename T, std::size_t Capacity, std::size_t MaxConsumers> class SpmcBroadcastDropRingBuffer {
  public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power-of-two.");
    static_assert(MaxConsumers > 0, "MaxConsumers must be > 0");

    SpmcBroadcastDropRingBuffer() : writer_cnt_(0), notify_seq_raw_(0), next_consumer_idx_(0), register_lock_(0) {
        for (auto &c : consumers_) {
            c.reader_cnt.store(0, std::memory_order_relaxed);
            c.active.store(false, std::memory_order_relaxed);
            c.id_hash.store(0, std::memory_order_relaxed);
            std::memset(c.id_str, 0, CONSUMER_ID_MAX_LEN);
        }

        // seq == 0 means "slot has never been published"
        for (auto &s : slots_) {
            s.seq.store(0, std::memory_order_relaxed);
        }
    }

    /**
     * Register a consumer with a unique string identifier.
     *
     * - If the identifier already exists, reuse that slot (reconnection)
     * - Otherwise, allocate a new slot
     * - Thread-safe: uses spinlock for registration
     *
     * @param id_str unique consumer identifier string
     * @param out_id output: assigned consumer index
     * @param start_from_latest if true (and new registration), consumer ignores
     * backlog
     * @return true if registration succeeded, false if no slots available
     */
    bool register_consumer(const char *id_str, std::size_t &out_id, bool start_from_latest = true) {
        if (id_str == nullptr || id_str[0] == '\0')
            return false;

        const uint64_t hash = hash_string(id_str);

        // Acquire lock with timeout so shutdown paths do not block forever.
        if (!acquire_register_lock("register_consumer")) {
            return false;
        }

        bool result = false;

        // 1) Search for existing consumer (reconnection handling)
        for (std::size_t i = 0; i < MaxConsumers; ++i) {
            if (consumers_[i].id_hash.load(std::memory_order_acquire) == hash &&
                std::strncmp(consumers_[i].id_str, id_str, CONSUMER_ID_MAX_LEN - 1) == 0) {
                // Existing consumer found - reactivate
                // Reset reader_cnt if start_from_latest is true or if reader_cnt >
                // writer_cnt (writer restarted)
                std::size_t w = writer_cnt_.load(std::memory_order_acquire);
                std::size_t r = consumers_[i].reader_cnt.load(std::memory_order_relaxed);
                if (start_from_latest || r > w) {
                    consumers_[i].reader_cnt.store(w, std::memory_order_relaxed);
                }
                consumers_[i].active.store(true, std::memory_order_release);
                out_id = i;
                result = true;
                goto unlock;
            }
        }

        // 2) Register new consumer - find empty slot
        {
            std::size_t idx = next_consumer_idx_.load(std::memory_order_relaxed);

            // Start searching from next_consumer_idx_
            for (std::size_t attempt = 0; attempt < MaxConsumers; ++attempt) {
                std::size_t candidate = (idx + attempt) % MaxConsumers;

                // Find inactive slot with empty ID
                if (!consumers_[candidate].active.load(std::memory_order_relaxed) &&
                    consumers_[candidate].id_hash.load(std::memory_order_relaxed) == 0) {

                    // Initialize slot
                    std::strncpy(consumers_[candidate].id_str, id_str, CONSUMER_ID_MAX_LEN - 1);
                    consumers_[candidate].id_str[CONSUMER_ID_MAX_LEN - 1] = '\0';
                    consumers_[candidate].id_hash.store(hash, std::memory_order_relaxed);

                    std::size_t w = writer_cnt_.load(std::memory_order_acquire);
                    consumers_[candidate].reader_cnt.store(start_from_latest ? w : oldest_available(w), std::memory_order_relaxed);
                    consumers_[candidate].active.store(true, std::memory_order_release);

                    // Update next index
                    next_consumer_idx_.store((candidate + 1) % MaxConsumers, std::memory_order_relaxed);

                    out_id = candidate;
                    result = true;
                    goto unlock;
                }
            }

            // 3) No empty slot available - reuse inactive slot (overwrite old ID)
            for (std::size_t i = 0; i < MaxConsumers; ++i) {
                if (!consumers_[i].active.load(std::memory_order_relaxed)) {
                    std::strncpy(consumers_[i].id_str, id_str, CONSUMER_ID_MAX_LEN - 1);
                    consumers_[i].id_str[CONSUMER_ID_MAX_LEN - 1] = '\0';
                    consumers_[i].id_hash.store(hash, std::memory_order_relaxed);

                    std::size_t w = writer_cnt_.load(std::memory_order_acquire);
                    consumers_[i].reader_cnt.store(start_from_latest ? w : oldest_available(w), std::memory_order_relaxed);
                    consumers_[i].active.store(true, std::memory_order_release);

                    out_id = i;
                    result = true;
                    goto unlock;
                }
            }
        }

    unlock:
        register_lock_.store(0, std::memory_order_release);
        return result;
    }

    /**
     * Unregister by string identifier.
     * Consumer state is preserved for potential reconnection.
     * Thread-safe: uses spinlock.
     */
    bool unregister_consumer(const char *id_str) {
        if (id_str == nullptr || id_str[0] == '\0')
            return false;

        const uint64_t hash = hash_string(id_str);

        // Acquire lock with timeout so shutdown paths do not block forever.
        if (!acquire_register_lock("unregister_consumer")) {
            return false;
        }

        bool result = false;
        for (std::size_t i = 0; i < MaxConsumers; ++i) {
            if (consumers_[i].id_hash.load(std::memory_order_relaxed) == hash &&
                std::strncmp(consumers_[i].id_str, id_str, CONSUMER_ID_MAX_LEN - 1) == 0) {
                consumers_[i].active.store(false, std::memory_order_release);
                // Preserve ID for state recovery on reconnection
                result = true;
                break;
            }
        }

        register_lock_.store(0, std::memory_order_release);
        return result;
    }

    /**
     * Find consumer index by string identifier.
     *
     * @return consumer index, or MaxConsumers if not found
     */
    std::size_t find_consumer(const char *id_str) const {
        if (id_str == nullptr || id_str[0] == '\0')
            return MaxConsumers;

        const uint64_t hash = hash_string(id_str);

        for (std::size_t i = 0; i < MaxConsumers; ++i) {
            if (consumers_[i].id_hash.load(std::memory_order_acquire) == hash &&
                std::strncmp(consumers_[i].id_str, id_str, CONSUMER_ID_MAX_LEN - 1) == 0) {
                return i;
            }
        }
        return MaxConsumers;
    }

    /**
     * Check if a consumer with the given ID is currently active.
     */
    bool is_consumer_active(const char *id_str) const {
        const uint64_t hash = hash_string(id_str);

        for (std::size_t i = 0; i < MaxConsumers; ++i) {
            if (consumers_[i].id_hash.load(std::memory_order_acquire) == hash &&
                std::strncmp(consumers_[i].id_str, id_str, CONSUMER_ID_MAX_LEN - 1) == 0) {
                return consumers_[i].active.load(std::memory_order_acquire);
            }
        }
        return false;
    }

    /**
     * Get the registered ID string for a consumer slot.
     */
    const char *get_consumer_id(std::size_t id) const {
        if (id >= MaxConsumers)
            return nullptr;
        return consumers_[id].id_str;
    }

    /**
     * Producer push.
     *
     * Always succeeds.
     * Overwrites old data when the ring wraps.
     *
     * If Linux futex notification is enabled, push() will wake waiters.
     */
    void push(const T &item) {
        std::size_t w   = writer_cnt_.load(std::memory_order_relaxed);
        std::size_t idx = w & (Capacity - 1);

        // 1) Write payload
        slots_[idx].data = item;

        // 2) Publish slot with sequence number (w + 1)
        //    seq == logical_index + 1
        slots_[idx].seq.store(w + 1, std::memory_order_release);

        // 3) Publish writer index
        writer_cnt_.store(w + 1, std::memory_order_release);

        // 4) Notify consumers (optional)
        notify_push();
    }

    /**
     * FIFO pull for a specific consumer (single-attempt, no recursion).
     *
     * Behavior:
     * - Reads at most ONE item.
     * - If the consumer is too slow, it jumps to the oldest available index
     *   (marks dropped=true) and returns false for this call.
     * - If the target slot is not yet published (seen < expected), returns false.
     * - If the target slot was overwritten (seen > expected), it resyncs to
     * oldest (dropped=true) and returns false.
     *
     * @param dropped set to true if the consumer missed data or was
     * resynchronized
     * @return true if exactly one item was read, false otherwise
     */
    bool pull(std::size_t id, T &out, bool &dropped) {
        dropped = false;
        if (id >= MaxConsumers)
            return false;
        if (!consumers_[id].active.load(std::memory_order_acquire))
            return false;

        std::size_t r = consumers_[id].reader_cnt.load(std::memory_order_relaxed);
        std::size_t w = writer_cnt_.load(std::memory_order_acquire);

        // No new data for this consumer
        if (r == w)
            return false;

        // If consumer is too slow, jump to the oldest available and stop (no read
        // this call)
        const std::size_t oldest = oldest_available(w);
        if (r < oldest) {
            consumers_[id].reader_cnt.store(oldest, std::memory_order_release);
            dropped = true;
            return false;
        }

        const std::size_t expected_seq = r + 1;
        const std::size_t idx          = r & (Capacity - 1);

        const std::size_t seen = slots_[idx].seq.load(std::memory_order_acquire);

        // Slot not yet published for this sequence -> do not spin; just report "not
        // ready"
        if (seen < expected_seq) {
            return false;
        }

        // Slot was overwritten -> resync to oldest and stop (no read this call)
        if (seen > expected_seq) {
            w = writer_cnt_.load(std::memory_order_acquire);
            consumers_[id].reader_cnt.store(oldest_available(w), std::memory_order_release);
            dropped = true;
            return false;
        }

        // seen == expected_seq: safe to read
        out = slots_[idx].data;
        consumers_[id].reader_cnt.store(r + 1, std::memory_order_release);
        return true;
    }

    /**
     * Read only the latest item and skip all intermediate ones.
     *
     * @param dropped true if any data was skipped
     * @param max_retries maximum retry attempts (default 1000)
     */
    bool pull_latest(std::size_t id, T &out, bool &dropped, std::size_t max_retries = 1000) {
        dropped = false;
        if (id >= MaxConsumers)
            return false;
        if (!consumers_[id].active.load(std::memory_order_acquire))
            return false;

        std::size_t w = writer_cnt_.load(std::memory_order_acquire);
        if (w == 0)
            return false;

        std::size_t r = consumers_[id].reader_cnt.load(std::memory_order_relaxed);
        if (r == w)
            return false;

        std::size_t target = w - 1;

        if (r < target)
            dropped = true;
        if (r < oldest_available(w))
            dropped = true;

        const std::size_t expected_seq = target + 1;

        for (std::size_t retry = 0; retry < max_retries; ++retry) {
            const std::size_t idx  = target & (Capacity - 1);
            const std::size_t seen = slots_[idx].seq.load(std::memory_order_acquire);

            if (seen == expected_seq) {
                out = slots_[idx].data;
                consumers_[id].reader_cnt.store(w, std::memory_order_release);
                return true;
            }

            if (seen < expected_seq) {
                // Slot not yet published, wait for it
                continue;
            }

            // Latest slot was overwritten again
            dropped = true;
            w       = writer_cnt_.load(std::memory_order_acquire);
            if (w == 0)
                return false;
            target = w - 1;
        }

        // Max retries exceeded - give up
        return false;
    }

    /**
     * View the latest published item (single-attempt, non-blocking).
     */
    bool view_latest(T &out, std::size_t &index) const {
        std::size_t w = writer_cnt_.load(std::memory_order_acquire);
        if (w == 0)
            return false;

        const std::size_t logical = w - 1;
        const std::size_t idx     = logical & (Capacity - 1);
        const std::size_t expect  = logical + 1;  // seq is logical_index + 1

        const std::size_t seen = slots_[idx].seq.load(std::memory_order_acquire);
        if (seen != expect)
            return false;

        out   = slots_[idx].data;
        index = idx;
        return true;
    }

    /**
     * Return the ring slot index (masked) of the latest item.
     */
    std::size_t view_latest_cnt() const {
        std::size_t w = writer_cnt_.load(std::memory_order_acquire);
        if (w == 0)
            return 0;
        return (w - 1) & (Capacity - 1);
    }

    std::size_t writer_cnt() const { return writer_cnt_.load(std::memory_order_acquire); }

    // ---------------------------
    // Notification API (Linux)
    // ---------------------------

    /**
     * Return the current notification sequence (32-bit).
     * A consumer can store this value and later wait until it changes.
     */
    uint32_t notify_seq() const { return notify_seq_raw_.load(std::memory_order_acquire); }

    /**
     * Wait until notify_seq changes from 'expected' (infinite wait).
     *
     * @return true if notify_seq changed (or changed before sleeping),
     *         false on non-linux build (no futex)
     */
    bool wait_for_push(uint32_t expected) const {
#ifdef __linux__
        // Fast-path: already changed
        if (notify_seq_raw_.load(std::memory_order_acquire) != expected)
            return true;

        // Sleep until woken (spurious wake-ups are allowed)
        futex_wait_u32(reinterpret_cast<const uint32_t *>(&notify_seq_raw_), expected);

        // After wake, check if it actually changed
        return notify_seq_raw_.load(std::memory_order_acquire) != expected;
#else
        (void)expected;
        return false;
#endif
    }

    /**
     * Wait until notify_seq changes from 'expected' OR timeout occurs.
     *
     * @param timeout_ns relative timeout in nanoseconds
     * @return true if notify_seq changed, false if timeout (or non-linux build)
     */
    bool wait_for_push(uint32_t expected, uint64_t timeout_ns) const {
#ifdef __linux__
        // Fast-path: already changed
        if (notify_seq_raw_.load(std::memory_order_acquire) != expected)
            return true;

        timespec ts;
        ts.tv_sec  = static_cast<time_t>(timeout_ns / 1000000000ull);
        ts.tv_nsec = static_cast<long>(timeout_ns % 1000000000ull);

        int rc = futex_wait_u32_timed(reinterpret_cast<const uint32_t *>(&notify_seq_raw_), expected, &ts);
        if (rc == 0) {
            return notify_seq_raw_.load(std::memory_order_acquire) != expected;
        }

        if (errno == ETIMEDOUT) {
            return false;
        }

        // EINTR/EAGAIN/spurious wake: re-check
        return notify_seq_raw_.load(std::memory_order_acquire) != expected;
#else
        (void)expected;
        (void)timeout_ns;
        return false;
#endif
    }

    /**
     * Wake consumers explicitly (e.g., for shutdown).
     * Increments notify_seq so waiting consumers detect the change.
     */
    void wake_all() {
#ifdef __linux__
        notify_seq_raw_.fetch_add(1, std::memory_order_release);
        futex_wake_u32(reinterpret_cast<const uint32_t *>(&notify_seq_raw_), INT_MAX);
#endif
    }

  private:
    // Liveness of a lock-holder pid via kill(pid, 0) (sends no signal): 0 = alive,
    // ESRCH = gone. EPERM (process exists but not ours) is treated as alive —
    // conservative; all igris_c processes run as the same user so EPERM is not
    // expected. Non-Linux lacks the check, so the steal path is disabled there.
    static bool pid_alive(uint32_t pid) {
#ifdef __linux__
        if (::kill(static_cast<pid_t>(pid), 0) == 0)
            return true;
        return errno != ESRCH;
#else
        (void)pid;
        return true;
#endif
    }

    static uint32_t self_pid() {
#ifdef __linux__
        return static_cast<uint32_t>(::getpid());
#else
        return 1u;  // sentinel; the steal path is Linux-only, so this value is unused
#endif
    }

    // Take the registration lock (stores our pid; 0 = free). If the current holder's
    // process has died mid critical-section, steal the lock — and log it — so a crash
    // can't brick register/unregister for every process. Bounded by a 2 s timeout
    // while the holder is alive.
    bool acquire_register_lock(const char *caller) {
        using namespace std::chrono;

        constexpr auto kTimeout      = seconds(2);
        constexpr auto kWarnInterval = milliseconds(250);
        constexpr auto kBackoff      = milliseconds(1);

        const uint32_t my_pid = self_pid();
        const auto start      = steady_clock::now();
        auto last_warn        = start;
        std::size_t spin_cnt  = 0;

        while (true) {
            uint32_t owner = 0;  // try to claim a free lock
            if (register_lock_.compare_exchange_strong(owner, my_pid, std::memory_order_acquire, std::memory_order_relaxed))
                return true;

            // `owner` now holds the current holder's pid. Steal it if that process is
            // gone (crashed without releasing); otherwise wait (bounded by kTimeout).
            if (owner != 0 && owner != my_pid && !pid_alive(owner)) {
                if (register_lock_.compare_exchange_strong(owner, my_pid, std::memory_order_acquire, std::memory_order_relaxed)) {
                    robros::shm::logger::get_logger("igris_c_shm")
                        .warn(std::string(caller) + ": register lock holder pid " + std::to_string(owner) + " is dead; stole the lock");
                    return true;
                }
                continue;  // another waiter stole it first — retry from the top
            }

            ++spin_cnt;
            const auto now = steady_clock::now();
            if (now - last_warn >= kWarnInterval) {
                robros::shm::logger::get_logger("igris_c_shm")
                    .info(std::string(caller) + ": waiting for register lock held by pid " + std::to_string(owner) +
                          " (spin=" + std::to_string(spin_cnt) + ")");
                last_warn = now;
            }
            if (now - start >= kTimeout) {
                robros::shm::logger::get_logger("igris_c_shm")
                    .error(std::string(caller) + ": timeout while waiting for register lock (held by pid " + std::to_string(owner) + ")");
                return false;
            }
            std::this_thread::sleep_for(kBackoff);
        }
    }

    /**
     * Simple string hash for fast comparison (FNV-1a variant).
     */
    static uint64_t hash_string(const char *str) {
        if (str == nullptr || str[0] == '\0')
            return 0;

        uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
        while (*str) {
            hash ^= static_cast<uint64_t>(*str++);
            hash *= 1099511628211ULL;  // FNV prime
        }
        return hash;
    }

    /**
     * Oldest logical index still available in the ring.
     */
    static std::size_t oldest_available(std::size_t w) { return (w > Capacity) ? (w - Capacity) : 0; }

    /**
     * Increment notify sequence and wake waiters (Linux futex).
     *
     * Using release ordering ensures that a woken consumer that does an acquire
     * load will observe the published slot data / writer_cnt_.
     */
    void notify_push() {
#ifdef __linux__
        notify_seq_raw_.fetch_add(1, std::memory_order_release);
        futex_wake_u32(reinterpret_cast<const uint32_t *>(&notify_seq_raw_), INT_MAX);
#endif
    }

    struct Slot {
        T data;
        std::atomic<std::size_t> seq;  // sequence number for safe publish
    };

    struct alignas(CACHELINE_SIZE) ConsumerState {
        std::atomic<std::size_t> reader_cnt;
        std::atomic<bool> active;
        std::atomic<uint64_t> id_hash;     // String hash for fast comparison
        char id_str[CONSUMER_ID_MAX_LEN];  // Unique identifier string
    };

  public:
    // Global writer counter (logical index)
    std::atomic<std::size_t> writer_cnt_;

    // 32-bit futex word in shared memory (atomic storage)
    // NOTE: futex operates on a 32-bit aligned word.
    alignas(4) mutable std::atomic<uint32_t> notify_seq_raw_;

  private:
    std::atomic<std::size_t> next_consumer_idx_;  // Next consumer index to allocate
    // Registration spinlock storing the holder's pid (0 = free). A holder that dies
    // mid critical-section is detected (kill(pid,0)==ESRCH) and its lock stolen, so a
    // crash can't brick register/unregister process-wide. uint32_t fits the prior
    // padding before slots_ → shm layout & size are unchanged (no ABI growth).
    std::atomic<uint32_t> register_lock_;
    std::array<Slot, Capacity> slots_;
    std::array<ConsumerState, MaxConsumers> consumers_;
};
}  // namespace robros::shm::buffer

#endif  // __SPMC_BROADCAST_DROP_RINGBUFFER_H__
