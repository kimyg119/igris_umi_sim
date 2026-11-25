#ifndef KEYBOARD
#define KEYBOARD

#include "igris_c_can/util/keyboard_typedef.h"

#include <errno.h>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <linux/input.h>
#include <thread>
#include <unistd.h>

namespace robros {
namespace hardware {
namespace keyboard {

struct KeyboardEventType_st {
    uint16_t type;
    KeyMap code;
    TrigType eventType;

    std::string toString() {
        std::stringstream sstr;
        sstr << "[TYPE]: " << type;
        sstr << "[CODE]: " << code.toInt();
        sstr << "[EVENT]: " << eventType.toString();
        return sstr.str();
    }
};

static KeyboardEventType_st inputEvent2KeyboardEvent(const input_event ie) {
    KeyboardEventType_st dst;
    dst.eventType = TrigType(ie.value);
    dst.type      = ie.type;
    dst.code      = ie.code;
    return dst;
}

typedef std::function<void(KeyboardEventType_st)> callback;

class Keyboard {
  public:
    Keyboard() = default;
#if (__cplusplus >= 201703L)
    Keyboard(std::string_view dev)
        : _dev(dev.data()), _callbackStop(false)
#else
    Keyboard(std::string dev)
        : _dev(dev), _callbackStop(false)
#endif
    {
        init(dev.data());
    }

    void init(std::string dev) {
        _dev = dev;
        _fd  = open(_dev.c_str(), O_RDONLY);

        if (_fd == -1)
            std::cout << "Could not open keyboard" << std::endl;
        else
            std::cout << "Keyboard opened" << std::endl;

        _th = std::thread([this] { rcvThread(); });
    }

    void exit() {
        if (close(_fd) == -1) {
            // close failed
            if (errno == EIO) {
            } else if (errno == EBADR) {
            }
            std::cout << "Xbox device close error" << std::endl;
        } else {
            std::cout << "Xbox device closed" << std::endl;
        }
    }

    void StopCallback() { _callbackStop = true; }

    void StartCallback() { _callbackStop = false; }

  private:
    bool _callbackStop;
    void eventCallback(input_event &event) {
        if (_callbackStop)
            return;
        auto keyboardEvent = inputEvent2KeyboardEvent(event);

        if (keyAllCallback)
            keyAllCallback(keyboardEvent);
        switch (keyboardEvent.eventType.toEvent()) {
        case TrigType::all:
            /* Do nothing */
            break;
        case TrigType::pressed:
            if (keyPressedCallback)
                keyPressedCallback(keyboardEvent);
            break;
        case TrigType::pushed:
            if (keyPushedCallback)
                keyPushedCallback(keyboardEvent);
            break;
        case TrigType::released:
            if (keyReleasedCallback)
                keyReleasedCallback(keyboardEvent);
            break;
        }
    }

  public:
    void rcvThread() {
        input_event ev;
        ssize_t n;

        while (1) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));

            if (_isStop == true)
                break;

            n = read(_fd, &ev, sizeof ev);

            if (n == (ssize_t)-1) {
                if (errno == EINTR)
                    continue;
                else
                    break;
            } else if (n != sizeof ev) {
                errno = EIO;
                break;
            }

            if (ev.type == EV_KEY) {
                eventCallback(ev);
            }
        }
        std::cout << "Keyboard closed" << std::endl;
    }

  public:
    void addCallback(callback callback, TrigType::Trig type) {
        switch (type) {
        case TrigType::all:
            keyAllCallback = callback;
            break;
        case TrigType::pressed:
            keyPressedCallback = callback;
            break;
        case TrigType::pushed:
            keyPushedCallback = callback;
            break;
        case TrigType::released:
            keyReleasedCallback = callback;
            break;
        }
    }

    int getFd() { return _fd; }

    bool isStop() { return _isStop; }

  private:
    std::string _dev;
    std::thread _th;
    int _fd      = -1;
    bool _isStop = false;

    callback keyPressedCallback, keyReleasedCallback, keyPushedCallback, keyAllCallback;
};
}  // namespace keyboard
}  // namespace hardware
}  // namespace robros

#endif /* KEYBOARD */
