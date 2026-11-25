#ifndef KEYBOARD_TYPEDEF
#define KEYBOARD_TYPEDEF

#include <sstream>

namespace robros {
namespace hardware {
namespace keyboard {
class KeyMap {
  public:
    enum Map {
        esc = 1,
        key1,
        key2,
        key3,
        key4,
        key5,
        key6,
        key7,
        key8,
        key9,
        key0,
        keyQ = 16,
        keyW,
        keyE,
        keyR,
        keyT,
        keyY,
        keyU,
        keyI,
        keyO,
        keyP,
        keySBRACKETS,
        keySBRACKETE,
        keyA = 30,
        keyS,
        keyD,
        keyF,
        keyG,
        keyH,
        keyJ,
        keyK,
        keyL,
        keySCOL,
        keySINGLEQUATATION,
        keyZ = 44,
        keyX,
        keyC,
        keyV,
        keyB,
        keyN,
        keyM,
        keyCOMMA,
        keyDOT,
        keySLASH
    };

    KeyMap() = default;
    KeyMap(int map) : _map(Map(map)) {}
    constexpr KeyMap(Map map) : _map(map) {}
    explicit operator bool() const = delete;

    constexpr bool operator==(KeyMap keyMap) const { return _map == keyMap._map; }

    constexpr bool operator==(Map map) const { return _map == map; }

    constexpr bool operator==(int map) const { return _map == Map(map); }

    constexpr bool operator!=(KeyMap keyMap) const { return _map != keyMap._map; }

    constexpr bool operator!=(Map map) const { return _map != map; }

    constexpr bool operator!=(int map) const { return _map != Map(map); }

    int toInt() { return static_cast<int>(_map); }
    Map toMap() { return _map; }

  private:
    Map _map;
};
class TrigType {
  public:
    enum Trig { released = 0, pressed, pushed, all };

    TrigType() = default;
    TrigType(int trig) : _trig(Trig(trig)) {}
    constexpr TrigType(Trig trig) : _trig(trig) {}
    explicit operator bool() const = delete;
    int toInt() { return static_cast<int>(_trig); }
    std::string toString() {
        std::stringstream sstr;
        switch (_trig) {
        case Trig::released:
            sstr << "released";
            break;
        case Trig::pressed:
            sstr << "puressed";
            break;
        case Trig::pushed:
            sstr << "pushed";
            break;
        case Trig::all:
            break;
        }
        return sstr.str();
    }
    Trig toEvent() { return _trig; }

  private:
    Trig _trig;
};
}  // namespace keyboard
}  // namespace hardware
}  // namespace robros

#endif /* KEYBOARD_TYPEDEF */
