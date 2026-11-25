/**
 * @file task.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2025-09-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef __TASK_A136318A_F83C_4FEC_B08C_3D193DC3547D_H__
#define __TASK_A136318A_F83C_4FEC_B08C_3D193DC3547D_H__

#include "igris_c_can/util/logger.h"

#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

class RtSingleTask {
  public:
    RtSingleTask(const char *name, std::function<void()> fn, int period_us, int priority = 70, int cpu = 0, int phase_us = 0,
                 clockid_t clock_id = CLOCK_MONOTONIC)
        : name_(name), fn_(std::move(fn)), period_ns_(int64_t(period_us) * 1000LL), priority_(priority), cpu_(cpu),
          phase_ns_(int64_t(phase_us) * 1000LL), clock_id_(clock_id) {}

    bool startAt(const timespec &start_ts, int period_us) {
        period_ns_ = int64_t(period_us) * 1000LL;
        return startAt(start_ts);
    }

    // Specify absolute start time (based on CLOCK_MONOTONIC)
    bool startAt(const timespec &start_ts) {
        stop_requested_.store(false, std::memory_order_relaxed);
        start_ts_ = start_ts;

        pthread_attr_t attr;
        pthread_attr_init(&attr);
        sched_param sch{};
        sch.sched_priority = priority_;
        pthread_attr_setschedpolicy(&attr, SCHED_FIFO);  // SCHED_OTHER
        pthread_attr_setschedparam(&attr, &sch);
        pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);

        if (pthread_create(&thread_, &attr, &RtSingleTask::trampoline, this) != 0) {
            perror("pthread_create");
            return false;
        }
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(cpu_, &cs);
        if (pthread_setaffinity_np(thread_, sizeof(cs), &cs) != 0)
            perror("pthread_setaffinity_np");
        return true;
    }

    bool startNow() {
        timespec now{};
        clock_gettime(clock_id_, &now);
        return startAt(now);
    }

    void stop() {
        stop_requested_.store(true, std::memory_order_relaxed);
        if (thread_)
            pthread_join(thread_, nullptr);
    }

    static void installSignalHandlers() {
        struct sigaction sa {};
        sa.sa_handler = &RtSingleTask::signalHandler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);
    }

  private:
    // --- Shared stop flag: all instances will stop on signal ---
    static inline std::atomic<bool> stop_all_{false};

    static void signalHandler(int signo) {
        if (signo == SIGINT || signo == SIGTERM)
            stop_all_.store(true, std::memory_order_relaxed);
    }

    static void *trampoline(void *arg) {
        reinterpret_cast<RtSingleTask *>(arg)->run();
        return nullptr;
    }

    static inline int64_t nsFromTs(const timespec &t) { return int64_t(t.tv_sec) * 1'000'000'000LL + t.tv_nsec; }

    static inline timespec tsFromNs(int64_t ns) {
        timespec r{.tv_sec = time_t(ns / 1'000'000'000LL), .tv_nsec = long(ns % 1'000'000'000LL)};
        if (r.tv_nsec < 0) {
            r.tv_nsec += 1'000'000'000L;
            r.tv_sec -= 1;
        }
        return r;
    }

    // Wait until the absolute target time (spin for the last 100µs)
    void waitUntil(int64_t target_ns) {
        static constexpr int64_t spin_ns = 100'000;  // 100µs
        timespec now{};
        clock_gettime(clock_id_, &now);
        int64_t now_ns = nsFromTs(now);

        int64_t sleep_until = target_ns - spin_ns;
        if (sleep_until > now_ns) {
            timespec ts = tsFromNs(sleep_until);
            int rc      = clock_nanosleep(clock_id_, TIMER_ABSTIME, &ts, nullptr);
            if (rc != 0 && rc != EINTR)
                robros::util::Logger::log(*robros::util::Logger::get(), spdlog::level::err,
                                          "[" + std::string(name_) + "] clock_nanosleep: " + strerror(rc));
        }
        while (true) {
            clock_gettime(clock_id_, &now);
            if (nsFromTs(now) >= target_ns)
                break;
            asm volatile("pause" ::: "memory");
        }
    }

    static int64_t alignNext(int64_t start_ns, int64_t period_ns, int64_t phase_ns, int64_t now_ns) {
        int64_t base = start_ns + phase_ns;
        if (now_ns <= base)
            return base;
        int64_t delta = now_ns - base;
        int64_t k     = (delta + period_ns - 1) / period_ns;  // ceil
        return base + k * period_ns;
    }

    void run() {
        int pol;
        sched_param sp{};
        pthread_getschedparam(pthread_self(), &pol, &sp);
        robros::util::Logger::log(*robros::util::Logger::get(), spdlog::level::info,
                                  "[" + std::string(name_) + "] RT policy=" + std::to_string(pol) +
                                      " prio=" + std::to_string(sp.sched_priority));

        // Align to start
        int64_t start_ns = nsFromTs(start_ts_);
        timespec now_ts{};
        clock_gettime(clock_id_, &now_ts);
        int64_t now_ns = nsFromTs(now_ts);

        if (period_ns_ <= 0)
            period_ns_ = 1;
        if (phase_ns_ < 0)
            phase_ns_ = 0;
        if (phase_ns_ >= period_ns_)
            phase_ns_ %= period_ns_;

        int64_t next_ns = alignNext(start_ns, period_ns_, phase_ns_, now_ns);
        waitUntil(next_ns);

        while (!stop_requested_.load(std::memory_order_relaxed) && !stop_all_.load(std::memory_order_relaxed)) {
            // do work
            fn_();

            // Align to the next period (catch-up in case of overrun)
            clock_gettime(clock_id_, &now_ts);
            now_ns = nsFromTs(now_ts);
            do {
                next_ns += period_ns_;
            } while (next_ns <= now_ns);
            waitUntil(next_ns);
        }

        robros::util::Logger::log(*robros::util::Logger::get(), spdlog::level::info, "[" + std::string(name_) + "] stopped");
    }

  private:
    const char *name_;
    std::function<void()> fn_;
    int64_t period_ns_;
    int priority_;
    int cpu_;
    int64_t phase_ns_;
    clockid_t clock_id_;
    timespec start_ts_{};
    pthread_t thread_{};
    std::atomic<bool> stop_requested_{false};
};

#endif /* __TASK_A136318A_F83C_4FEC_B08C_3D193DC3547D_H__ */
