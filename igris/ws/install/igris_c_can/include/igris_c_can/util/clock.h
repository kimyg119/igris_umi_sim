#ifndef __IGRIS_C_CAN_INCLUDE_UTIL_CLOCK_H__
#define __IGRIS_C_CAN_INCLUDE_UTIL_CLOCK_H__
#include <chrono>
#include <iostream>
#include <sstream>

namespace robros {
namespace util {

static uint64_t getTick() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_nsec / 1000000) + ((uint64_t)ts.tv_sec * 1000ull);
}

static uint64_t microTick() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_nsec / 1000) + ((uint64_t)ts.tv_sec * 1000000ull);
}

struct tick_st {
    uint64_t nowTick;
    uint64_t pastTick;
    tick_st() {
        update();
        reset();
    }

    /**
     * @brief call getTick() method.
     */
    void update() { nowTick = getTick(); }

    /**
     * @brief reset tick.
     */
    void reset() { pastTick = nowTick; }

    /**
     * @brief Calculate the difference between nowTick and pastTick.
     *
     * @return uint64_t
     */
    uint64_t diff() { return nowTick - pastTick; }

    /**
     * @brief Call tick_st::update() and tick_st::diff() sequencely.
     *
     * @return uint64_t
     */
    uint64_t updateDiff() {
        update();
        return diff();
    }

    /**
     * @brief Generate data with std::string type.
     *
     * @return std::string
     */
    std::string toString() {
        std::stringstream sstr;
        sstr << "[tick]";
        sstr << " nowTick: " << nowTick;
        sstr << ", pastTick: " << pastTick;
        sstr << ", diff: " << diff();
        return sstr.str();
    }

    /**
     * @brief Compare time tick.
     *
     * @param tick
     * @return true
     * @return false
     */
    bool operator>(int32_t tick) { return updateDiff() > tick; }

    /**
     * @brief Compare time tick.
     *
     * @param tick
     * @return true
     * @return false
     */
    bool operator<(int32_t tick) { return updateDiff() < tick; }

    /**
     * @brief Compare the time tick (millisecond).
     *
     * @param millisecond
     * @return true
     * @return false
     */
    bool operator>(std::chrono::milliseconds millisecond) { return updateDiff() > millisecond.count(); }
};

struct microTick_st {
    uint64_t nowTick;
    uint64_t pastTick;
    microTick_st() {
        update();
        reset();
    }

    void update() { nowTick = microTick(); }

    void reset() { pastTick = nowTick; }

    uint64_t diff() { return nowTick - pastTick; }

    uint64_t updateDiff() {
        update();
        return diff();
    }

    std::string toString() {
        std::stringstream sstr;
        sstr << "[microtick]";
        sstr << " nowTick: " << nowTick;
        sstr << ", pastTick: " << pastTick;
        sstr << ", diff: " << diff();
        return sstr.str();
    }
    bool operator>(uint64_t tick) { return updateDiff() > tick; }
    bool operator<(uint64_t tick) { return updateDiff() < tick; }
    bool operator>(std::chrono::microseconds microsecond) { return updateDiff() > microsecond.count(); }
};

}  // namespace util
}  // namespace robros

#endif /* __IGRIS_C_CAN_INCLUDE_UTIL_CLOCK_H__ */
