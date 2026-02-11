#ifndef ADC_CONTINUOUS_H_
#define ADC_CONTINUOUS_H_

#include "sensor_config.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>

#define BUFFER_SIZE 16  // Message buffer size
#define ADC_GET_DATA(p_data) ((p_data)->type1.data)

void init_gpio(void);
QueueHandle_t get_message_buffer(void);
void read_adc(void *pvParameters);
TaskHandle_t* get_adc_task_handle(void);

void calibrate_dac2(); 
uint16_t (*get_compensation_table())[COLUMN_NUM];
void test_single_point(void); 

#endif // ADC_CONTINUOUS_H_
