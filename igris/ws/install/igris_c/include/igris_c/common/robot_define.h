#ifndef __ROBOT_DEFINE_AEEF015A_3188_46EA_9097_4CCDB8A6608F_H__
#define __ROBOT_DEFINE_AEEF015A_3188_46EA_9097_4CCDB8A6608F_H__

#include <iostream>
#include <string>
#include <vector>

#define JOINT_NUM 31
#define MODEL_DOF 31
#define CAN_CHANNEL_NUM 4
#define ENDEFFECTOR_NUMBER 2
#define LINK_NUMBER 32
#define MODEL_DOF_VIRTUAL 37
#define MODEL_DOF_QVIRTUAL 38

static const char link_name[LINK_NUMBER][50] = {
    "",
};

static const char joint_name[MODEL_DOF][50] = {
    "waist_yaw",          "waist_roll",        "waist_pitch",          "l_hip_pitch",
    "l_hip_roll",      "l_hip_yaw",      "l_knee_pitch",      "l_ankle_pitch",
    "l_ankle_roll",    "r_hip_pitch",   "r_hip_roll",       "r_hip_yaw",
    "r_knee_pitch",   "r_ankle_pitch", "r_ankle_roll",     "l_shoulder_pitch",
    "l_shoulder_roll", "l_shoulder_yaw", "l_elbow_pitch",     "l_wrist_yaw",
    "l_wrist_roll",    "l_wrist_pitch",  "r_shoulder_pitch", "r_shoulder_roll",
    "r_shoulder_yaw", "r_elbow_pitch", "r_wrist_yaw",      "r_wrist_roll",
    "r_wrist_pitch",  "neck_yaw",          "neck_pitch",
};

enum joint_idx {
    waist_yaw = 0,
    waist_roll,
    waist_pitch,
    l_hip_pitch,
    l_hip_roll,
    l_hip_yaw,
    l_knee_pitch,
    l_ankle_pitch,
    l_ankle_roll,
    r_hip_pitch,
    r_hip_roll,
    r_hip_yaw,
    r_knee_pitch,
    r_ankle_pitch,
    r_ankle_roll,
    l_shoulder_pitch,
    l_shoulder_roll,
    l_shoulder_yaw,
    l_elbow_pitch,
    l_wrist_yaw,
    l_wrist_roll,
    l_wrist_pitch,
    r_shoulder_pitch,
    r_shoulder_roll,
    r_shoulder_yaw,
    r_elbow_pitch,
    r_wrist_yaw,
    r_wrist_roll,
    r_wrist_pitch,
    neck_yaw,
    neck_pitch,
};

#endif /* __ROBOT_DEFINE_AEEF015A_3188_46EA_9097_4CCDB8A6608F_H__ */
