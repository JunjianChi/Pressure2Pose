/* imu_icm45686.h */
#pragma once
#include "esp_err.h"

#ifdef __cplusplus
extern "C" { 
#endif

typedef struct {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
} imu_raw_t;

esp_err_t imu_icm45686_init(void);
esp_err_t imu_icm45686_get_latest_raw(imu_raw_t *out);

#ifdef __cplusplus
}
#endif
