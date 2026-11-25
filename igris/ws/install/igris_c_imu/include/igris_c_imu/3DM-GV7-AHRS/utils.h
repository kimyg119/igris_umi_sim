#ifndef __UTILS_E7C15A00_0EA9_481F_8F8D_D35D34052A39_H__
#define __UTILS_E7C15A00_0EA9_481F_8F8D_D35D34052A39_H__

// #include <mip/definitions/data_filter.h>
#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double w;
    double x;
    double y;
    double z;
} Quaternion;

typedef struct {
    double x;
    double y;
    double z;
} Vector3;

Quaternion quaternion_multiply(Quaternion q1, Quaternion q2);
Quaternion quaternion_conjugate(Quaternion q);
Quaternion quaternion_normalize(Quaternion q);
Quaternion quat_ned_to_enu(const float w, const float x, const float y, const float z);
// Vector3 quaternion_to_euler(Quaternion q);
Vector3 quaternion_to_euler(const float w, const float x, const float y, const float z);
Vector3 quaternion_to_euler_xyz(const float w, const float x, const float y, const float z);
Vector3 rotate_vector_by_quaternion(Vector3 v, Quaternion q);

// Displays current filter state for the connected device if it has changed.
// void displayFilterState(const mip_filter_mode filterState, char **currentState, bool isFiveSeries);
void process_filter_flag(uint16_t data);

void bias_uncertainty_to_covariance(const float bias_uncertainty[3], double covariance[9]);

#ifdef __cplusplus
}
#endif

#endif /* __UTILS_E7C15A00_0EA9_481F_8F8D_D35D34052A39_H__ */
