/* imu_icm45686.c – ICM‑45686 minimal wrapper for ESP‑IDF --------------------------------
 *  • SPI4 10 MHz, INT1 watermark=2 frames, stream FIFO
 *  • gives two public APIs:  imu_icm45686_init()  &  imu_icm45686_get_latest_raw()
 *  -----------------------------------------------------------------------------*/
#include <stdbool.h>
#include <string.h>
#include "esp_attr.h"
#include "esp_log.h"
#include "imu_icm45686.h"
#include "imu/inv_imu_driver_advanced.h"
#include "system_interface.h"

#define TAG         "imu45686"
#define SERIF_TYPE  UI_SPI4

static inv_imu_device_t dev;
static uint8_t fifo_buf[FIFO_MIRRORING_SIZE];  /* per driver header: 4128 bytes */           /* big enough for many frames */
static volatile bool    int1_flag;               /* set in GPIO ISR */

/* latest raw cache (updated in sensor_event_cb) */
static imu_raw_t        latest_raw;
static volatile bool    raw_ready;

/* -------------------------- GPIO INT1 ISR -----------------------------------*/
static void IRAM_ATTR int1_cb(void *ctx, unsigned int num)
{
    (void)ctx; (void)num;
    int1_flag = true;
}

/* -------------------------- Sensor event cb ---------------------------------*/
static void sensor_event_cb(inv_imu_sensor_event_t *ev)
{
    /* keep only entries containing both accel & gyro */
    if (!(ev->sensor_mask & (1 << INV_SENSOR_ACCEL)) ||
        !(ev->sensor_mask & (1 << INV_SENSOR_GYRO)))
        return;

    latest_raw.ax = ev->accel[0]; latest_raw.ay = ev->accel[1]; latest_raw.az = ev->accel[2];
    latest_raw.gx = ev->gyro[0];  latest_raw.gy = ev->gyro[1];  latest_raw.gz = ev->gyro[2];
    raw_ready     = true;
}

/* --------------------------- MCU peripherals --------------------------------*/
static int init_mcu_low_level(void)
{
    int rc = 0;
    rc |= si_board_init();
    rc |= si_init_timers();
    rc |= si_io_imu_init(SERIF_TYPE);
    rc |= si_init_gpio_int(SI_GPIO_INT1, int1_cb);
    rc |= si_config_uart_for_print(SI_UART_ID_FTDI, INV_MSG_LEVEL_INFO);
    return rc;
}

/* --------------------------- public: init -----------------------------------*/
esp_err_t imu_icm45686_init(void)
{
    int rc = init_mcu_low_level();
    if (rc) return ESP_FAIL;

    /* hook transport */
    dev.transport.read_reg   = si_io_imu_read_reg;
    dev.transport.write_reg  = si_io_imu_write_reg;
    dev.transport.serif_type = SERIF_TYPE;
    dev.transport.sleep_us   = si_sleep_us;

    /* ----- soft-reset  + WHOAMI check ------------------------------------ */
    uint8_t who = 0x00;
    si_io_imu_read_reg(0x72, &who, 1);   // ← 0x72 是 ICM-45686 的 WHO_AM_I
    ESP_LOGI(TAG, "WHO_AM_I read 0x%02X", who);
    if (who != 0xE9) {                   // ← 正确 ID 应为 0xE9
        ESP_LOGE(TAG, "Unexpected WHO_AM_I");
        return ESP_FAIL;
    }

    inv_imu_int_pin_config_t pin_cfg = {
               .int_polarity = INTX_CONFIG2_INTX_POLARITY_HIGH,
               .int_mode     = INTX_CONFIG2_INTX_MODE_PULSE,
               .int_drive    = INTX_CONFIG2_INTX_DRIVE_PP,
            };
            rc |= inv_imu_set_pin_config_int(&dev, INV_IMU_INT1, &pin_cfg);
        
            inv_imu_int_state_t int_cfg = { 0 };
            int_cfg.INV_FIFO_THS = INV_IMU_ENABLE;      // FIFO 达到 watermark 触发 INT1
            rc |= inv_imu_set_config_int(&dev, INV_IMU_INT1, &int_cfg);

            
    /* init driver */
    rc = inv_imu_adv_init(&dev);
    if (rc) { ESP_LOGE(TAG, "driver init fail %d", rc); return ESP_FAIL; }

    /* register sensor event callback */
    inv_imu_adv_var_t *adv = (inv_imu_adv_var_t *)dev.adv_var;
    adv->sensor_event_cb = sensor_event_cb;

    /* FIFO config */
    inv_imu_adv_fifo_config_t fifo;
    rc |= inv_imu_adv_get_fifo_config(&dev, &fifo);
    fifo.base_conf.accel_en   = 1;
    fifo.base_conf.gyro_en    = 1;
    fifo.base_conf.hires_en   = 0;
    fifo.base_conf.fifo_wm_th = 2;                              /* AN‑000364 */
    fifo.base_conf.fifo_mode  = FIFO_CONFIG0_FIFO_MODE_STREAM;
    fifo.tmst_fsync_en        = INV_IMU_ENABLE;
    fifo.fifo_wr_wm_gt_th     = FIFO_CONFIG2_FIFO_WR_WM_EQ_OR_GT_TH;
    rc |= inv_imu_adv_set_fifo_config(&dev, &fifo);

    /* FSR & ODR */
    rc |= inv_imu_set_accel_fsr(&dev,  ACCEL_CONFIG0_ACCEL_UI_FS_SEL_4_G);
    rc |= inv_imu_set_gyro_fsr (&dev,  GYRO_CONFIG0_GYRO_UI_FS_SEL_2000_DPS);
    rc |= inv_imu_set_accel_frequency(&dev, ACCEL_CONFIG0_ACCEL_ODR_200_HZ);
    rc |= inv_imu_set_gyro_frequency (&dev, GYRO_CONFIG0_GYRO_ODR_200_HZ);

    /* enable sensors (Low‑Noise) */
    rc |= inv_imu_adv_enable_accel_ln(&dev);
    rc |= inv_imu_adv_enable_gyro_ln (&dev);

    if (rc) { ESP_LOGE(TAG, "IMU cfg error %d", rc); return ESP_FAIL; }

    raw_ready = false;
    int1_flag = false;
    return ESP_OK;
}

/* --------------------------- public: read -----------------------------------*/
// Interrupt version: pop latest frame only when INT1 is triggered, not working hh
// esp_err_t imu_icm45686_get_latest_raw(imu_raw_t *out)
// {
//     if (!int1_flag)                           /* no watermark yet */
//         return ESP_ERR_INVALID_STATE;

//     int1_flag = false;

//     uint16_t bytes = 0;
//     if (inv_imu_adv_get_data_from_fifo(&dev, fifo_buf, &bytes))
//         return ESP_FAIL;

//     if (inv_imu_adv_parse_fifo_data(&dev, fifo_buf, bytes))
//         return ESP_FAIL;

//     if (!raw_ready)
//         return ESP_ERR_INVALID_STATE;

//     raw_ready = false;                        /* consume */
//     memcpy(out, &latest_raw, sizeof(imu_raw_t));
//     return ESP_OK;
// }


// Polling version: pop latest frame regardless of INT
esp_err_t imu_icm45686_get_latest_raw_poll(imu_raw_t *out)
{
    uint16_t cnt = 0;
    if (inv_imu_get_frame_count(&dev, &cnt) || cnt == 0) {
        return ESP_ERR_INVALID_STATE;
    }
    uint16_t bytes = 0;
    if (inv_imu_adv_get_data_from_fifo(&dev, fifo_buf, &bytes)) {
        return ESP_FAIL;
    }
    if (inv_imu_adv_parse_fifo_data(&dev, fifo_buf, bytes)) {
        return ESP_FAIL;
    }
    if (!raw_ready) {
        return ESP_ERR_INVALID_STATE;
    }
    raw_ready = false;
    memcpy(out, &latest_raw, sizeof(*out));
    return ESP_OK;

    // uint16_t cnt = 0;
    // if (inv_imu_get_frame_count(&dev, &cnt) || cnt == 0) {
    //     return ESP_ERR_INVALID_STATE;
    // }

    // // —— 限制一次只读 1 帧 ——  
    // const uint16_t FRAME_BYTES = sizeof(int16_t)*6;   // accel+gyro 各 3 轴 * 2 bytes
    // uint16_t frames_to_read = 1;                      // 只读 1 帧
    // uint16_t bytes = frames_to_read * FRAME_BYTES;

    // // 拉出一帧数据
    // if (inv_imu_adv_get_data_from_fifo(&dev, fifo_buf, &bytes)) {
    //     return ESP_FAIL;
    // }
    // if (inv_imu_adv_parse_fifo_data(&dev, fifo_buf, bytes)) {
    //     return ESP_FAIL;
    // }

    // if (!raw_ready) {
    //     return ESP_ERR_INVALID_STATE;
    // }
    // raw_ready = false;
    // memcpy(out, &latest_raw, sizeof(*out));
    // return ESP_OK;
}

// Original API name for compatibility
esp_err_t imu_icm45686_get_latest_raw(imu_raw_t *out)
{
    return imu_icm45686_get_latest_raw_poll(out);
}
