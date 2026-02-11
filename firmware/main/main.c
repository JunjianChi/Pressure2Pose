#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/message_buffer.h"
#include "wifi_sender.h"
#include "adc_continuous.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"
#include "esp_task_wdt.h"
#include "sensor_config.h"
#include "dac_continuous.h"
#include "imu/inv_imu_driver.h"
#include "system_interface.h" 
#include "imu_icm45686.h" 

bool adc_task_status = false;
bool wifi_task_status = false;

void app_main(void) {
    // close watchdog
    esp_task_wdt_deinit();

    // Initialize NVS must be done before any other functions from nvs_flash
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    // Initialize GPIO for MUX conntrol
    init_gpio();

    // // Initialize WiFi
    wifi_init_sta();
    udp_connection_init();

    init_dac();

    #if CONFIG_IMU_ENABLED
        ESP_ERROR_CHECK(imu_icm45686_init());      
    #endif


    // // call main function
    xTaskCreatePinnedToCore(read_adc, "read_adc_task", 1024 * 80, NULL, 2, get_adc_task_handle(), tskNO_AFFINITY);
}
