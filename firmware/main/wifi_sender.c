#include "wifi_sender.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/err.h"
#include "lwip/sys.h"
#include "lwip/netdb.h"
#include "sensor_config.h"
#include "adc_continuous.h"
#include "freertos/queue.h"
#include "esp_timer.h"

EventGroupHandle_t wifi_event_group;
const int WIFI_CONNECTED_BIT = BIT0;
volatile bool wifi_connected = false;
volatile bool udp_connected = false;

static TaskHandle_t s_task_handle = NULL;
static struct sockaddr_in dest_addr;

char addr_str[128];
int addr_family;
int ip_protocol;
volatile int sock = -1;

TaskHandle_t* get_wifi_task_handle(void) {
    return &s_task_handle;
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_connected = false;
        udp_connected = false;
        sock = -1;
        xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
        esp_wifi_connect();
        ESP_LOGI(TAG, "Disconnected from WiFi, retrying...");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
        wifi_connected = true;
        ESP_LOGI(TAG, "Connected to WiFi");
    }
}

bool wifi_is_connected(void) {
    return wifi_connected;
}

bool udp_is_connected(void) {
    return udp_connected;
}

void wifi_init_sta(void) {
    wifi_event_group = xEventGroupCreate();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    
    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &instance_got_ip));
    
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(ESP_IF_WIFI_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Wait until connected
    xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);

    ESP_LOGI(TAG, "wifi_init_sta finished.");
}

int udp_connection_init(void) {
    // Initialize the event group
    while (1) {
        if (!wifi_is_connected()) {
            vTaskDelay(1000 / portTICK_PERIOD_MS);
            continue;
        } else {
            dest_addr.sin_addr.s_addr = inet_addr(SERVICE_ADDRESS);
            dest_addr.sin_family = AF_INET;
            dest_addr.sin_port = htons(SERVICE_PORT);

            inet_ntoa_r(dest_addr.sin_addr, addr_str, sizeof(addr_str) - 1);
            addr_family = AF_INET;
            ip_protocol = IPPROTO_IP;

            sock = socket(addr_family, SOCK_DGRAM, ip_protocol);
            if (sock < 0) {
                ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
                vTaskDelay(1000 / portTICK_PERIOD_MS);  // Delay before retrying
                continue;
            }
            udp_connected = true;
            ESP_LOGI(TAG, "Socket created, sending to %s:%d", addr_str, SERVICE_PORT);
            return sock;  // Return the socket descriptor for further use
        }
    }
}



// static int packet_count = 0;
// static int interval = 35;
// static int64_t start_time = 0;


int send_udp_sock(int sock_get, const void *data, size_t size) {
    if (sock_get < 0) {
        ESP_LOGE(TAG, "Socket not created, errno %d", errno);
        return -1;
    }
    if (!udp_connected) {
        ESP_LOGE(TAG, "UDP not connected, errno %d", errno);
        return -1;
    }
    // wait until able to send
    while (1) {
        int len = sendto(sock_get, data, size, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        if (len < 0) {
            // ESP_LOGE(TAG, "Error occurred during sending: errno %d", errno);
            esp_rom_delay_us(10);
            continue;
        }
        return len;
    }
}

int send_udp(const void *data, size_t size) {
    if (sock < 0) {
        ESP_LOGE(TAG, "Socket not created, errno %d", errno);
        return -1;
    }
    if (!udp_connected) {
        ESP_LOGE(TAG, "UDP not connected, errno %d", errno);
        return -1;
    }

    // if (packet_count == 0) {
    //     start_time = esp_timer_get_time();
    // }

    // wait until able to send
    while (1) {
        int len = sendto(sock, data, size, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        if (len < 0) {
            // ESP_LOGE(TAG, "Error occurred during sending: errno %d", errno);
            // esp_rom_delay_us(10);
            continue;
        }

        // packet_count++;

        // if (packet_count >= interval) {
        //     int64_t end_time = esp_timer_get_time();
        //     int64_t elapsed_time = end_time - start_time;
        //     ESP_LOGI(TAG, "Sent %d packets in %lld microseconds", interval, elapsed_time);
        //     packet_count = 0;
        // }
        return len;
    }
}
