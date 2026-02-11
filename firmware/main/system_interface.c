/* system_interface.c – ESP‑IDF board glue for ICM‑45686 driver
 * provides si_*  functions expected by InvenSense examples           */
#include "system_interface.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/portmacro.h"
#include <string.h>
#include "esp_rom_sys.h"   


/* ---------- user config pins from sdkconfig ---------------------------- */
#ifndef IMU_CS
#define IMU_CS    CONFIG_IMU_CS
#endif
#ifndef IMU_SCLK
#define IMU_SCLK  CONFIG_IMU_SCLK
#endif
#ifndef IMU_MOSI
#define IMU_MOSI  CONFIG_IMU_MOSI
#endif
#ifndef IMU_MISO
#define IMU_MISO  CONFIG_IMU_MISO
#endif
#ifndef IMU_INT
#define IMU_INT   CONFIG_IMU_INT
#endif

#define TAG "sys_if"

static spi_device_handle_t s_dev;

/* ---------------------------------------------------------------------- */
int si_board_init(void) { return ESP_OK; }
int si_init_timers(void) { return ESP_OK; }

/* ---------------- SPI bus --------------------------------------------- */
int si_io_imu_init(inv_imu_serif_type_t type)
{
    if (type != UI_SPI4) return -1;
    spi_bus_config_t bus = {
        .mosi_io_num = IMU_MOSI,
        .miso_io_num = IMU_MISO,
        .sclk_io_num = IMU_SCLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4096*2,
    };
     esp_err_t err = spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO);
    if (err && err != ESP_ERR_INVALID_STATE) return -1;

    spi_device_interface_config_t devcfg = {
        .clock_speed_hz = 10 * 1000 * 1000,
        .mode = 3,
        .spics_io_num = IMU_CS,
        .queue_size = 1,
        .flags = SPI_DEVICE_NO_DUMMY,
    };
    return spi_bus_add_device(SPI2_HOST, &devcfg, &s_dev);
}

/* helper */
static int spi_txrx(const uint8_t *tx, uint8_t *rx, size_t len)
{
    spi_transaction_t t = {0};
    t.length = len * 8;

    if (tx) {
        t.tx_buffer = tx;
    }
    if (rx) {
        t.rx_buffer = rx;
        t.rxlength = t.length;           /* for half‑duplex RX‑only */
    }
    return spi_device_polling_transmit(s_dev, &t);
}

int si_io_imu_write_reg(uint8_t reg, const uint8_t *buf, uint32_t len)
{
    uint8_t tmp[len + 1];
    tmp[0] = reg & 0x7F;
    memcpy(&tmp[1], buf, len);
    return (spi_txrx(tmp, NULL, len + 1) == ESP_OK) ? 0 : -1;
}

int si_io_imu_read_reg(uint8_t reg, uint8_t *buf, uint32_t len)
{
    uint8_t tx[len + 1];
    uint8_t rx[len + 1];
    tx[0] = reg | 0x80;                   /* bit7=1 → read */
    memset(&tx[1], 0, len);               /* dummy bytes */

    spi_transaction_t t = {
        .length    = (len + 1) * 8,      
        .tx_buffer = tx,
        .rx_buffer = rx,
    };
    if (spi_device_polling_transmit(s_dev, &t) != ESP_OK)
        return -1;

    memcpy(buf, &rx[1], len);             
    return 0;
}

/* ---------------------------------------------------------------------- */
void si_sleep_us(uint32_t us) { esp_rom_delay_us(us); }
uint64_t si_get_time_us(void) { return esp_timer_get_time(); }

/* ---------------- INT GPIO -------------------------------------------- */
static void (*s_gpio_cb)(void*, unsigned int);
static void IRAM_ATTR gpio_isr(void *arg)
{
    if (s_gpio_cb) s_gpio_cb(NULL, (unsigned int)(uintptr_t)arg);
}

int si_init_gpio_int(unsigned int num, void (*cb)(void*, unsigned int))
{
    s_gpio_cb = cb;
    gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << IMU_INT,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .intr_type = GPIO_INTR_POSEDGE,
    };
    if (gpio_config(&cfg) != ESP_OK) return -1;
    gpio_install_isr_service(0);
    return gpio_isr_handler_add(IMU_INT, gpio_isr, (void*)(uintptr_t)num);
}

/* ---------------- critical section ------------------------------------ */
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
void si_disable_irq(void) { portENTER_CRITICAL(&s_mux); }
void si_enable_irq(void)  { portEXIT_CRITICAL(&s_mux);  }

/* ---------------- UART stubs ------------------------------------------ */
int si_config_uart_for_print(inv_uart_num_t id, int level)
{ (void)id; esp_log_level_set(TAG, level); return 0; }
int si_get_uart_command(inv_uart_num_t id, char *c)
{ (void)id; (void)c; return 0; }
int si_config_uart_for_bin(inv_uart_num_t id)
{ (void)id; return 0; }

/* ---------------- log funnel ------------------------------------------ */
void inv_msg(int lvl, const char *fmt, ...)
{
    esp_log_level_t lv = (lvl==0)?ESP_LOG_ERROR:(lvl==1)?ESP_LOG_ERROR:(lvl==2)?ESP_LOG_WARN:ESP_LOG_INFO;
    va_list ap; va_start(ap, fmt);
    esp_log_writev(lv, TAG, fmt, ap);
    va_end(ap);
}