#ifndef __COMMON_C3B77EC3_A156_4E94_9B41_9F7E0C707AA6_H__
#define __COMMON_C3B77EC3_A156_4E94_9B41_9F7E0C707AA6_H__

#include "igris_c_can/pcan/PCANBasic.h"

#include <iostream>
#include <string>

void dispCanMessage(TPCANMsg msg, std::string prefix) {
    std::cout << prefix << " message: ";
    std::cout << "ID: " << std::hex << msg.ID << ", " << "Length: " << (int)msg.LEN << ", " << "Data: ";
    for (int i = 0; i < msg.LEN; ++i) {
        std::cout << std::hex << (int)msg.DATA[i] << " ";
    }
    std::cout << std::dec << std::endl;
}

#endif /* __COMMON_C3B77EC3_A156_4E94_9B41_9F7E0C707AA6_H__ */
