#ifndef _SENSOR_CONFIG_H_
#define _SENSOR_CONFIG_H_

// Include the sdkconfig header
#include "sdkconfig.h"
#include "esp_adc/adc_continuous.h"

#define WR01 CONFIG_WR01
#define WR02 CONFIG_WR02
#define A0   CONFIG_A0
#define A1   CONFIG_A1
#define A2   CONFIG_A2
#define A3   CONFIG_A3
#define A4   CONFIG_A4
#define IA_A0 CONFIG_IA_A0
#define IA_A1 CONFIG_IA_A1
#define IA_A2 CONFIG_IA_A2
#define IA_CS CONFIG_IA_CS
#define IMU_CS CONFIG_IMU_CS
#define IMU_SCLK CONFIG_IMU_SCLK
#define IMU_MISO CONFIG_IMU_MISO
#define IMU_MOSI CONFIG_IMU_MOSI
#define IMU_INT CONFIG_IMU_INT
#define IMU_ENABLED CONFIG_IMU_ENABLED
#define IMU_DISABLED CONFIG_IMU_DISABLED

#if CONFIG_FOOT_LEFT
    #define FOOT_SIDE 0x0000   /* Left foot → 0x0000 */
#elif CONFIG_FOOT_RIGHT
    #define FOOT_SIDE 0x0001   /* Right foot → 0x0001 */
#else
    #error "Must select FOOT_LEFT or FOOT_RIGHT in menuconfig"
#endif

#if CONFIG_MAT_SENSOR
    #define MAT_SENSOR
    #define TAG  "esp32_sensor_mat"
    // if not staic it should be defined in the main.c and extern in the header file
#endif

#if CONFIG_INSOLE_SENSOR
    #define INSOLE_SENSOR
    #define TAG  "esp32_sensor_insole"
#endif

#define ADC_CHANNEL    CONFIG_ADC_CHANNEL
#define SAMPLE_RATE    (200*1000)
#define COLUMN_NUM     CONFIG_SENSOR_COL_SIZE
#define ROW_NUM        CONFIG_SENSOR_ROW_SIZE
#define AVERAGE_NUM    CONFIG_ADC_AVERAGE_NUM
// #define AVERAGE_NUM    4
#define READ_LEN       (AVERAGE_NUM * SOC_ADC_DIGI_RESULT_BYTES)



#if CONFIG_DATA_ENCODE_ASCII
    #define DATA_ENCODE_ASCII
    #define ADC_DATA_SIZE_IN_ASCII 4
    // Type 1
    #if CONFIG_DATA_PACKAGE_COLUMN
        #define COLUMN_HEADER_SIZE 4
        #define PAYLOAD_SIZE (ROW_NUM * ADC_DATA_SIZE_IN_ASCII + COLUMN_HEADER_SIZE)
    #endif

    // Type 2
    #if CONFIG_DATA_PACKAGE_FRAME
        #define PAYLOAD_SIZE (ROW_NUM * COLUMN_NUM * ADC_DATA_SIZE_IN_ASCII)
    #endif
#endif

#if CONFIG_DATA_ENCODE_RAW
    #define DATA_ENCODE_RAW

    // Type 3
    #if CONFIG_DATA_PACKAGE_COLUMN
        #define COLUMN_HEADER_SIZE 4
        #define PAYLOAD_SIZE (ROW_NUM * SOC_ADC_DIGI_RESULT_BYTES + COLUMN_HEADER_SIZE)
    #endif

    // Type 4
    #if CONFIG_DATA_PACKAGE_FRAME && CONFIG_INSOLE_SENSOR && CONFIG_SINGLE_FRAME && CONFIG_IMU_DISABLED
        #define PAYLOAD_SIZE (254)
        #define IDX_FOOT     253
    #endif

    #if CONFIG_DATA_PACKAGE_FRAME && CONFIG_INSOLE_SENSOR && CONFIG_SINGLE_FRAME && CONFIG_IMU_ENABLED
        #define PAYLOAD_SIZE (254+6)
        #define IDX_FOOT     253
        #define IDX_AX       254
        #define IDX_AY       255
        #define IDX_AZ       256
        #define IDX_GX       257
        #define IDX_GY       258
        #define IDX_GZ       259
    #endif

    #if CONFIG_DATA_PACKAGE_FRAME && CONFIG_INSOLE_SENSOR && CONFIG_DOUBLE_FRAME && CONFIG_IMU_DISABLED
        #define PAYLOAD_SIZE (507)
        #define IDX_FOOT     506
    #endif

    #if CONFIG_DATA_PACKAGE_FRAME && CONFIG_INSOLE_SENSOR && CONFIG_DOUBLE_FRAME && CONFIG_IMU_ENABLED
        #define PAYLOAD_SIZE (507 + 6)
        #define IDX_FOOT     506
        #define IDX_AX       507
        #define IDX_AY       508
        #define IDX_AZ       509
        #define IDX_GX       510
        #define IDX_GY       511
        #define IDX_GZ       512
    #endif

    #if CONFIG_DATA_PACKAGE_FRAME && CONFIG_MAT_SENSOR
        #define PAYLOAD_SIZE (ROW_NUM * COLUMN_NUM + 1)
    #endif

#endif

#endif  // _SENSOR_CONFIG_H_
