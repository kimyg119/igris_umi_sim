/**
 * @file rollingBuffer.h
 * @author colson (colson@robros.co.kr)
 * @brief
 * @version 0.1
 * @date 2026-01-03
 *
 * @copyright Copyright (c) 2026
 *
 */

#ifndef __ROLLINGBUFFER_AE8EE3F4_8AD8_48D4_ABA9_2619F8409AAE_H__
#define __ROLLINGBUFFER_AE8EE3F4_8AD8_48D4_ABA9_2619F8409AAE_H__

#include <Eigen/Dense>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <vector>

namespace robros::shm::buffer {

template <typename T> class RollingBuffer {
  public:
    RollingBuffer() = default;
    RollingBuffer(size_t buff_size) : max_size_(buff_size) {}
    RollingBuffer(size_t buff_size, const T &init_value) { init(buff_size, init_value); }

    void init(size_t buff_size, const T &init_value) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.clear();
        buffer_.resize(buff_size, init_value);
        max_size_ = buff_size;

        // push once
        buffer_.push_back(init_value);
        if (buffer_.size() > max_size_) {
            buffer_.erase(buffer_.begin());
        }
    }
    void push_back(const T &value) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.push_back(value);
        if (buffer_.size() > max_size_) {
            buffer_.erase(buffer_.begin());
        }
    }
    const T &get(size_t index) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= buffer_.size()) {
            throw std::out_of_range("RollingBuffer: index out of range");
        }
        return buffer_[index];
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

    // Convert buffer to a concatenated Eigen::VectorXd
    Eigen::VectorXd toEigenVector() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.empty())
            return Eigen::VectorXd();

        const size_t single_dim = buffer_[0].size();
        Eigen::VectorXd result(single_dim * buffer_.size());

        for (size_t i = 0; i < buffer_.size(); ++i) {
            result.segment(i * single_dim, single_dim) = buffer_[i];
        }
        return result;
    }

  private:
    std::vector<T> buffer_;
    size_t max_size_ = 0;
    mutable std::mutex mutex_;
};
}  // namespace robros::shm::buffer
#endif /* __ROLLINGBUFFER_AE8EE3F4_8AD8_48D4_ABA9_2619F8409AAE_H__ */
