#include "adc_continuous.h"
#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_err.h"
#include "driver/gpio.h"
#include "esp_rom_sys.h"
#include "freertos/queue.h"
#include "wifi_sender.h"
#include "dac_continuous.h"
#include "system_interface.h"
#include "imu/inv_imu_driver.h"
#include "imu_icm45686.h" 

// ADC configuration
static adc_continuous_handle_t handle = NULL;
// adc_continuous_handle_t handle = NULL;
TaskHandle_t s_task_handle = NULL;

static adc_channel_t channel[1] = {ADC_CHANNEL};

QueueHandle_t messageBuffer = NULL;

TaskHandle_t* get_adc_task_handle(void) {
    return &s_task_handle;
}

#if (CONFIG_INSOLE_SENSOR)
// Pattern to remove redundant index (253 points from 32*10 matrix)
int pattern[ROW_NUM][COLUMN_NUM] = {
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0},  
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0},  
    {1, 1, 1, 1, 1, 1, 1, 0, 0, 0},  
    {1, 1, 1, 1, 1, 1, 1, 1, 0, 0},   
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 0},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 0},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 0},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 0},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 0},  
    {1, 1, 1, 1, 1, 1, 1, 1, 0, 0},  
    {1, 1, 1, 1, 1, 1, 1, 1, 0, 0},  
    {1, 1, 1, 1, 1, 1, 1, 0, 0, 0},  
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0}, 
    {1, 1, 1, 1, 1, 1, 1, 0, 0, 0},  
    {1, 1, 1, 1, 1, 1, 1, 0, 0, 0},  
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0},  
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0},  
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0},  
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0},  
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0}, 
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0}, 
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0}, 
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}  
};
#endif

#if (CONFIG_CUR_SRC)
    static uint16_t compensation_table[ROW_NUM][COLUMN_NUM] = {0}; // Store DAC2 calibrated value
    static bool is_calibrated = false; // Mark if calibration is done
#endif

void init_gpio(void) {
    gpio_config_t io_conf;
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = (1ULL << WR01) | (1ULL << WR02) | (1ULL << A0) | (1ULL << A1) | (1ULL << A2) | (1ULL << A3) | (1ULL << A4) | (1ULL << IA_A0) | (1ULL << IA_A1) | (1ULL << IA_A2) | (1ULL << IA_CS);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;

    gpio_config(&io_conf);

    // Disable Mux
    gpio_set_level(WR01, 1);
    gpio_set_level(WR02, 1);
}

void selectMuxChannel(uint8_t wrPin, uint8_t channel) {
    if (channel > 31) return;  // 0-31 channel

    // Enable Mux by grounding wrPin
    gpio_set_level(wrPin, 0);
    // esp_rom_delay_us(100);

    // Channel selection
    gpio_set_level(A0, (channel & 0x01));
    gpio_set_level(A1, (channel & 0x02) >> 1);
    gpio_set_level(A2, (channel & 0x04) >> 2);
    gpio_set_level(A3, (channel & 0x08) >> 3);
    gpio_set_level(A4, (channel & 0x10) >> 4);

    // Disable Mux by pulling up wrPin
    gpio_set_level(wrPin, 1);
    // esp_rom_delay_us(100);
}

void selectIAGain(uint8_t gain){
    gpio_set_level(IA_CS, 0); 

    if (gain == 1){
        gpio_set_level(IA_A0, 0);
        gpio_set_level(IA_A1, 0);
        gpio_set_level(IA_A2, 0);
    } else if (gain == 2){
        gpio_set_level(IA_A0, 1);
        gpio_set_level(IA_A1, 0);
        gpio_set_level(IA_A2, 0);
    } else if (gain == 4){
        gpio_set_level(IA_A0, 0);
        gpio_set_level(IA_A1, 1);
        gpio_set_level(IA_A2, 0);
    } else if (gain == 8){
        gpio_set_level(IA_A0, 1);
        gpio_set_level(IA_A1, 1);
        gpio_set_level(IA_A2, 0);
    } else if (gain == 16){
        gpio_set_level(IA_A0, 0);
        gpio_set_level(IA_A1, 0);
        gpio_set_level(IA_A2, 1);
    } else if (gain == 32){
        gpio_set_level(IA_A0, 1);
        gpio_set_level(IA_A1, 0);
        gpio_set_level(IA_A2, 1);
    } else if (gain == 64){
        gpio_set_level(IA_A0, 0);
        gpio_set_level(IA_A1, 1);
        gpio_set_level(IA_A2, 1);
    } else if (gain == 128){
        gpio_set_level(IA_A0, 1);
        gpio_set_level(IA_A1, 1);
        gpio_set_level(IA_A2, 1);
    }
}

static bool IRAM_ATTR s_conv_done_cb(adc_continuous_handle_t handle, const adc_continuous_evt_data_t *edata, void *user_data) {
    BaseType_t mustYield = pdFALSE;

    // ADC conversion task complete
    vTaskNotifyGiveFromISR(s_task_handle, &mustYield);

    return (mustYield == pdTRUE);
}

static void continuous_adc_init(adc_channel_t *channel, uint8_t channel_num, adc_continuous_handle_t *out_handle) {
    adc_continuous_handle_t handle = NULL;

    // Configure the initialization structure for continuous ADC conversion
    adc_continuous_handle_cfg_t adc_config = {
        .max_store_buf_size = SOC_ADC_DIGI_RESULT_BYTES * AVERAGE_NUM * 2,  // Max buffer size
        .conv_frame_size = READ_LEN,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_config, &handle));

    adc_continuous_config_t dig_cfg = {
        .sample_freq_hz = SAMPLE_RATE,
        .conv_mode = ADC_CONV_SINGLE_UNIT_1,
        .format = ADC_DIGI_OUTPUT_FORMAT_TYPE1,
    };

    // ADC mode and channel setting
    adc_digi_pattern_config_t adc_pattern[SOC_ADC_PATT_LEN_MAX] = {0};
    dig_cfg.pattern_num = channel_num;
    for (int i = 0; i < channel_num; i++) {
        adc_pattern[i].atten = ADC_ATTEN_DB_12;
        adc_pattern[i].channel = channel[i] & 0x7;
        adc_pattern[i].unit = ADC_UNIT_1;
        adc_pattern[i].bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;

        ESP_LOGI(TAG, "adc_pattern[%d].atten is :%"PRIx8, i, adc_pattern[i].atten);
        ESP_LOGI(TAG, "adc_pattern[%d].channel is :%"PRIx8, i, adc_pattern[i].channel);
        ESP_LOGI(TAG, "adc_pattern[%d].unit is :%"PRIx8, i, adc_pattern[i].unit);
    }
    dig_cfg.adc_pattern = adc_pattern;
    ESP_ERROR_CHECK(adc_continuous_config(handle, &dig_cfg));

    *out_handle = handle;
}


void read_adc(void *pvParameters) {
    esp_err_t ret;
    uint32_t ret_num = 0;
    uint8_t result[READ_LEN] = {0}; // ADC read buffer
    
    memset(result, 0xcc, READ_LEN);
    
    #if (CONFIG_DATA_ENCODE_RAW && CONFIG_DATA_PACKAGE_FRAME) || (CONFIG_DATA_ENCODE_ASCII && CONFIG_DATA_PACKAGE_FRAME)
        uint16_t payload[PAYLOAD_SIZE] = {0}; // ADC send buffer
        int num_samples_read = 0;
    #endif

    #if (CONFIG_DATA_ENCODE_ASCII && CONFIG_DATA_PACKAGE_COLUMN) || (CONFIG_DATA_ENCODE_RAW && CONFIG_DATA_PACKAGE_COLUMN)
        char payload[PAYLOAD_SIZE] = {0}; // ADC send buffer
    #endif
    uint32_t sum = 0;

    // Get task handle
    s_task_handle = xTaskGetCurrentTaskHandle();

    // Initialize continuous ADC
    continuous_adc_init(channel, sizeof(channel) / sizeof(adc_channel_t), &handle);

    adc_continuous_evt_cbs_t cbs = {
        .on_conv_done = s_conv_done_cb, 
    };
    ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(handle, &cbs, NULL)); 
    ESP_ERROR_CHECK(adc_continuous_start(handle));

    #if CONFIG_DATA_ENCODE_ASCII && CONFIG_DATA_PACKAGE_COLUMN
        payload[0] = 'c';
        payload[1] = 'o';
    #endif

    #if CONFIG_DATA_ENCODE_RAW && CONFIG_DATA_PACKAGE_COLUMN
        payload[0] = 'c';
        payload[1] = 'o';
    #endif

    while (1) {
        // Wait for the ADC conversion task to complete
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        #if (CONFIG_DATA_ENCODE_RAW && CONFIG_DATA_PACKAGE_FRAME) || (CONFIG_DATA_ENCODE_ASCII && CONFIG_DATA_PACKAGE_FRAME)
                num_samples_read = 0;
        #endif

        #if (CONFIG_INSOLE_SENSOR)
            #if CONFIG_DATA_ENCODE_RAW && CONFIG_DATA_PACKAGE_FRAME && CONFIG_POT_DIV
                // Read and process ADC data
                num_samples_read = 0;
                for (int col = 0; col < COLUMN_NUM; col++) {
                        selectMuxChannel(WR02, col);
                        for (int row = 0; row < ROW_NUM; row++){
                            if (pattern[row][col] == 1){
                                selectMuxChannel(WR01, row);
                                // esp_rom_delay_us(100);
                                sum = 0; // Reset sum before each row read
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);
                                // esp_rom_delay_us(100);
                                for (int i = 0; i < AVERAGE_NUM; i++) {
                                    adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i * SOC_ADC_DIGI_RESULT_BYTES];
                                    int data = ADC_GET_DATA(p);
                                    sum += (data & 0xFFFF);
                                }
                                payload[num_samples_read] = (uint16_t)(sum / AVERAGE_NUM) & 0xFFFF;
                                num_samples_read++;  
                                // ESP_LOGI(TAG, "Payload[%d]: %x", row,(uint16_t)(sum / AVERAGE_NUM) & 0xFFFF);
                            }
                        }
                    }

                    payload[253] = 0x0000; // Assign left (0x0000) or right (0x0001)
                    
                    if (wifi_is_connected() && udp_is_connected()) {
                        send_udp(payload, PAYLOAD_SIZE* sizeof(uint16_t));
                    }
            #endif

            #if CONFIG_DATA_ENCODE_RAW && CONFIG_DATA_PACKAGE_FRAME && CONFIG_CUR_SRC
                
                #if CONFIG_SINGLE_FRAME && CONFIG_IMU_DISABLED
                    selectIAGain(1);
                    num_samples_read = 0;
                    set_dac_voltage(DAC1_CHANNEL, 3300);
                    set_dac_voltage(DAC2_CHANNEL, 1600); 

                    for (int col = 0; col < COLUMN_NUM; col++) {
                        selectMuxChannel(WR02, col);
                        for (int row = 0; row < ROW_NUM; row++) {
                            if (pattern[row][col] == 1) { 
                                selectMuxChannel(WR01, row);
                                // dummy read
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);
                                // Real read
                                sum = 0;
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);

                                for (int i = 0; i < AVERAGE_NUM; i++) {
                                    adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i * SOC_ADC_DIGI_RESULT_BYTES];
                                    sum += (ADC_GET_DATA(p) & 0xFFFF);
                                }

                                payload[num_samples_read] = (uint16_t)(sum / AVERAGE_NUM) & 0xFFFF;
                                // ESP_LOGI(TAG, "Payload[%d]: %x", row,(uint16_t)(sum / AVERAGE_NUM) & 0xFFFF);
                                num_samples_read++;
                            }
                        }
                    }
                    payload[253] = 0x0000;
                    // if (wifi_is_connected() && udp_is_connected()) {
                    //     send_udp(payload, PAYLOAD_SIZE * sizeof(uint16_t));
                    // }
                #endif
                
                #if CONFIG_SINGLE_FRAME && CONFIG_IMU_ENABLED
                    selectIAGain(1);
                    num_samples_read = 0;
                    set_dac_voltage(DAC1_CHANNEL, 3300);
                    set_dac_voltage(DAC2_CHANNEL, 1600); 

                    for (int col = 0; col < COLUMN_NUM; col++) {
                        selectMuxChannel(WR02, col);
                        for (int row = 0; row < ROW_NUM; row++) {
                            if (pattern[row][col] == 1) { 
                                selectMuxChannel(WR01, row);
                                // dummy read
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);
                                // Real read
                                sum = 0;
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);

                                for (int i = 0; i < AVERAGE_NUM; i++) {
                                    adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i * SOC_ADC_DIGI_RESULT_BYTES];
                                    sum += (ADC_GET_DATA(p) & 0xFFFF);
                                }

                                payload[num_samples_read] = (uint16_t)(sum / AVERAGE_NUM) & 0xFFFF;
                                // ESP_LOGI(TAG, "Payload[%d]: %x", row,(uint16_t)(sum / AVERAGE_NUM) & 0xFFFF);
                                num_samples_read++;
                            }
                        }
                    }

                    payload[IDX_FOOT] = FOOT_SIDE; 

                    // IMU data
                    imu_raw_t imu;
                    if (imu_icm45686_get_latest_raw(&imu) == ESP_OK) {
                        payload[IDX_AX] = (uint16_t)imu.ax;
                        payload[IDX_AY] = (uint16_t)imu.ay;
                        payload[IDX_AZ] = (uint16_t)imu.az;
                        payload[IDX_GX] = (uint16_t)imu.gx;
                        payload[IDX_GY] = (uint16_t)imu.gy;
                        payload[IDX_GZ] = (uint16_t)imu.gz;

                        ESP_LOGI(TAG, "IMU Data: AX: %04x AY: %04x AZ: %04x GX: %04x GY: %04x GZ: %04x",
                            (uint16_t)imu.ax, (uint16_t)imu.ay, (uint16_t)imu.az,
                            (uint16_t)imu.gx, (uint16_t)imu.gy, (uint16_t)imu.gz);

                    } else {                        
                        memset(&payload[IDX_AX], 0, 6 * sizeof(uint16_t));
                    }
                    
                    if (wifi_is_connected() && udp_is_connected()) {
                        send_udp(payload, PAYLOAD_SIZE* sizeof(uint16_t));
                    }
                #endif


                #if CONFIG_DOUBLE_FRAME && CONFIG_IMU_DISABLED

                    
                    // Small gain for the first frame
                    selectIAGain(1);
                    num_samples_read = 0;
                    set_dac_voltage(DAC1_CHANNEL, 3300);
                    // set_dac_voltage(DAC2_CHANNEL, 750); 

                    set_dac_voltage(DAC2_CHANNEL, 1600); 

                    for (int col = 0; col < COLUMN_NUM; col++) {
                        selectMuxChannel(WR02, col);

                        for (int row = 0; row < ROW_NUM; row++) {
                            if (pattern[row][col] == 1) { 
                                selectMuxChannel(WR01, row);
                                // dummy read
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);

                                sum = 0;
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);

                                for (int i = 0; i < AVERAGE_NUM; i++) {
                                    adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i * SOC_ADC_DIGI_RESULT_BYTES];
                                    sum += (ADC_GET_DATA(p) & 0xFFFF);
                                }

                                payload[num_samples_read] = (uint16_t)(sum / AVERAGE_NUM) & 0xFFFF;
                                // ESP_LOGI(TAG, "Payload[%d]: %x", row,(uint16_t)(sum / AVERAGE_NUM) & 0xFFFF);
                                num_samples_read++;
                            }
                        }
                    }


                // High gain for the second frame
                    selectIAGain(16);
                    num_samples_read = 253;
                    set_dac_voltage(DAC1_CHANNEL, 3300);
                    set_dac_voltage(DAC2_CHANNEL, 750); 

                    // set_dac_voltage(DAC2_CHANNEL, 1600); 

                    for (int col = 0; col < COLUMN_NUM; col++) {
                        selectMuxChannel(WR02, col);

                        for (int row = 0; row < ROW_NUM; row++) {
                            if (pattern[row][col] == 1) { 
                                selectMuxChannel(WR01, row);
                                // dummy read
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);
                                // set_dac_voltage(DAC2_CHANNEL, comp_table[row][col]);

                                sum = 0;
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);

                                for (int i = 0; i < AVERAGE_NUM; i++) {
                                    adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i * SOC_ADC_DIGI_RESULT_BYTES];
                                    sum += (ADC_GET_DATA(p) & 0xFFFF);
                                }

                                payload[num_samples_read] = (uint16_t)(sum / AVERAGE_NUM) & 0xFFFF;
                                // ESP_LOGI(TAG, "Payload[%d]: %x", row,(uint16_t)(sum / AVERAGE_NUM) & 0xFFFF);
                                num_samples_read++;
                            }
                        }
                    }
                    payload[506] = FOOT_LEFT; 
                    if (wifi_is_connected() && udp_is_connected()) {
                        send_udp(payload, PAYLOAD_SIZE* sizeof(uint16_t));
                    }
                #endif

                #if CONFIG_DOUBLE_FRAME && CONFIG_IMU_ENABLED
                    selectIAGain(1);
                    num_samples_read = 0;
                    set_dac_voltage(DAC1_CHANNEL, 3300);
                    set_dac_voltage(DAC2_CHANNEL, 1600); 

                    for (int col = 0; col < COLUMN_NUM; col++) {
                        selectMuxChannel(WR02, col);
                        for (int row = 0; row < ROW_NUM; row++) {
                            if (pattern[row][col] == 1) { // **仅读取有效传感器**
                                selectMuxChannel(WR01, row);
                                // dummy read
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);
                                // Real read
                                sum = 0;
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);

                                for (int i = 0; i < AVERAGE_NUM; i++) {
                                    adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i * SOC_ADC_DIGI_RESULT_BYTES];
                                    sum += (ADC_GET_DATA(p) & 0xFFFF);
                                }

                                payload[num_samples_read] = (uint16_t)(sum / AVERAGE_NUM) & 0xFFFF;
                                // ESP_LOGI(TAG, "Payload[%d]: %x", row,(uint16_t)(sum / AVERAGE_NUM) & 0xFFFF);
                                num_samples_read++;
                            }
                        }
                    }

                    // High gain for the second frame
                    selectIAGain(16);
                    num_samples_read = 253;
                    set_dac_voltage(DAC1_CHANNEL, 3300);
                    set_dac_voltage(DAC2_CHANNEL, 750); 

                    for (int col = 0; col < COLUMN_NUM; col++) {
                        selectMuxChannel(WR02, col);
                        for (int row = 0; row < ROW_NUM; row++) {
                            if (pattern[row][col] == 1) { 
                                selectMuxChannel(WR01, row);
                                // dummy read
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);
                                // real read
                                sum = 0;
                                do {
                                    ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                                } while (ret != ESP_OK);

                                for (int i = 0; i < AVERAGE_NUM; i++) {
                                    adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i * SOC_ADC_DIGI_RESULT_BYTES];
                                    sum += (ADC_GET_DATA(p) & 0xFFFF);
                                }

                                payload[num_samples_read] = (uint16_t)(sum / AVERAGE_NUM) & 0xFFFF;
                                // ESP_LOGI(TAG, "Payload[%d]: %x", row,(uint16_t)(sum / AVERAGE_NUM) & 0xFFFF);
                                num_samples_read++;
                            }
                        }
                    }

                    // // **WiFi 传输数据**
                    payload[IDX_FOOT] = FOOT_SIDE; // 左侧（0x0000）或右侧（0x0001）

                    // IMU data
                    imu_raw_t imu;
                    if (imu_icm45686_get_latest_raw(&imu) == ESP_OK) {
                        payload[IDX_AX] = (uint16_t)imu.ax;
                        payload[IDX_AY] = (uint16_t)imu.ay;
                        payload[IDX_AZ] = (uint16_t)imu.az;
                        payload[IDX_GX] = (uint16_t)imu.gx;
                        payload[IDX_GY] = (uint16_t)imu.gy;
                        payload[IDX_GZ] = (uint16_t)imu.gz;

                        // ESP_LOGI(TAG, "IMU Data: AX: %04x AY: %04x AZ: %04x GX: %04x GY: %04x GZ: %04x",
                        //     (uint16_t)imu.ax, (uint16_t)imu.ay, (uint16_t)imu.az,
                        //     (uint16_t)imu.gx, (uint16_t)imu.gy, (uint16_t)imu.gz);

                    } else {                        
                        memset(&payload[IDX_AX], 0, 6 * sizeof(uint16_t));
                    }
                    
                    if (wifi_is_connected() && udp_is_connected()) {
                        send_udp(payload, PAYLOAD_SIZE* sizeof(uint16_t));
                    }
                #endif
            #endif 
        #endif // CONFIG_INSOLE_SENSOR

        #if (CONFIG_MAT_SENSOR)
            #if CONFIG_DATA_ENCODE_RAW && CONFIG_DATA_PACKAGE_COLUMN
                    // Read and process ADC data
                    for (int col = 0; col < COLUMN_NUM; col++) {
                        selectMuxChannel(WR01, col);
                        for (int row = 0; row < ROW_NUM; row++) {
                            sum = 0; // Reset sum before each row read
                            selectMuxChannel(WR02, ROW_NUM - 1 - row);
                            esp_rom_delay_us(50);
                            do {
                                ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                            } while (ret != ESP_OK);
                            for (int i = 0; i < AVERAGE_NUM; i++) {
                                adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i * SOC_ADC_DIGI_RESULT_BYTES];
                                int data = ADC_GET_DATA(p);
                                sum += (data & 0xFFFF);
                            }
                            // If RAW encoding is enabled, send the raw um / AVERAGE_NUM
                            uint16_t average = (uint16_t)(sum / AVERAGE_NUM);
                            payload[row * SOC_ADC_DIGI_RESULT_BYTES + COLUMN_HEADER_SIZE] = average & 0xFF;
                        }
                        payload[2] = '0' + col / 10;
                        payload[3] = '0' + col % 10;
                        // Add a newline to the end of the payload
                        payload[PAYLOAD_SIZE] = '\n';
                        // ESP_LOGI(TAG, "Payload: %s", payload);
                    }
            #endif

            #if CONFIG_DATA_ENCODE_RAW && CONFIG_DATA_PACKAGE_FRAME && CONFIG_POT_DIV
            num_samples_read = 0;
            for (int col = 0; col < COLUMN_NUM; col++) {
                    selectMuxChannel(WR02, col);
                    // esp_rom_delay_us(100);
                    for (int row = 0; row < ROW_NUM; row++){
                        selectMuxChannel(WR01, row);
                        // esp_rom_delay_us(100);
                        sum = 0; // Reset sum before each row read
                        do {
                            ret = adc_continuous_read(handle, result, READ_LEN, &ret_num, 0);
                        } while (ret != ESP_OK);
                        // esp_rom_delay_us(100);
                        for (int i = 0; i < AVERAGE_NUM; i++) {
                            adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i * SOC_ADC_DIGI_RESULT_BYTES];
                            int data = ADC_GET_DATA(p);
                            sum += (data & 0xFFFF);
                        }
                        payload[num_samples_read] = (uint16_t)(sum / AVERAGE_NUM) & 0xFFFF;
                        num_samples_read++;  
                        // ESP_LOGI(TAG, "Payload[%d]: %x", row,(uint16_t)(sum / AVERAGE_NUM) & 0xFFFF);
                        
                    }
                }

                payload[ROW_NUM*COLUMN_NUM] = 0x0000; // Assign left (0x0000) or right (0x0001)           
                
                if (wifi_is_connected() && udp_is_connected()) {
                    send_udp(payload, PAYLOAD_SIZE* sizeof(uint16_t));
                }
            #endif 
        #endif // CONFIG_MAT_SENSOR
    } // end of while loop
}// end of read_adc


