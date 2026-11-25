/**
 * @file time.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-12-30
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __TIME_BFD68C2E_A7A6_4E3E_8FA4_A7A4C4694C5F_H__
#define __TIME_BFD68C2E_A7A6_4E3E_8FA4_A7A4C4694C5F_H__
#include <chrono>
#include <iostream>
#include <sstream>

namespace robros::util::time {

inline uint64_t get_timestamp_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

template <typename Resolution> uint64_t getTick() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    if constexpr (std::is_same_v<Resolution, std::chrono::milliseconds>) {
        return (uint64_t)(ts.tv_nsec / 1000000) + ((uint64_t)ts.tv_sec * 1000ull);
    } else if constexpr (std::is_same_v<Resolution, std::chrono::microseconds>) {
        return (uint64_t)(ts.tv_nsec / 1000) + ((uint64_t)ts.tv_sec * 1000000ull);
    } else {
        static_assert("Unsupported Resolution type for getTick()");
        return 0;
    }
}

template <typename Resolution> struct tick_st {
    uint64_t nowTick;
    uint64_t pastTick;

    tick_st() {
        update();
        reset();
    }
    void update() { nowTick = getTick<Resolution>(); }
    void reset() { pastTick = nowTick; }
    uint64_t diff() { return nowTick - pastTick; }
    uint64_t updateDiff() {
        update();
        return diff();
    }

    bool operator>(uint64_t tick) { return updateDiff() > tick; }
    bool operator<(uint64_t tick) { return updateDiff() < tick; }
    bool operator>(Resolution duration) { return updateDiff() > duration.count(); }

    std::string toString() {
        std::stringstream sstr;
        if constexpr (std::is_same_v<Resolution, std::chrono::milliseconds>) {
            sstr << "[tick]";
        } else if constexpr (std::is_same_v<Resolution, std::chrono::microseconds>) {
            sstr << "[microtick]";
        }
        sstr << " nowTick: " << nowTick;
        sstr << ", pastTick: " << pastTick;
        sstr << ", diff: " << diff();
        return sstr.str();
    }
};

// Type aliases for convenience
using MilliTick = tick_st<std::chrono::milliseconds>;
using MicroTick = tick_st<std::chrono::microseconds>;
}  // namespace robros::util::time

#endif /* __TIME_BFD68C2E_A7A6_4E3E_8FA4_A7A4C4694C5F_H__ */
