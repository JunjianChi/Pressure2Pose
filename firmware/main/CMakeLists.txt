idf_component_register(
    SRCS
        "main.c"
        "adc_continuous.c"
        "dac_continuous.c"
        "wifi_sender.c"

        # 你的 IMU 包装层
        "imu_icm45686.c"

        # InvenSense driver – ★不要带空格、不要重复
        "imu/inv_imu_driver.c"
        "imu/inv_imu_driver_advanced.c"
        "imu/inv_imu_transport.c"          # ← 新建的桥接文件，见下面
        "system_interface.c"

    INCLUDE_DIRS "."
)
