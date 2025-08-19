# AutoPilot-G503-PS25: Autonomous Rover Project

This project contains the software for an autonomous rover platform based on a Jetson Nano and Pixhawk. The current implementation focuses on a real-time video streaming and communication system using MQTT.

## Core Components

### `publisher.py`
This script is designed to run on the rover. It captures live video from a webcam, encodes it into a compressed format, and publishes it over the network to an MQTT broker.

### `subscriber.py`
This script is designed to run on a monitoring computer. It connects to the MQTT broker, subscribes to the video stream topic, and displays the received video feed in real-time.

## How to Run

1.  **Setup the Broker:** Install and run a Mosquitto MQTT broker on a machine accessible by both the publisher and subscriber.
2.  **Update IPs:** In both scripts, update the `BROKER` IP address to the address of the machine running Mosquitto.
3.  **Run the Subscriber:** Start the `subscriber.py` script on the monitoring PC.
    ```bash
    python subscriber.py
    ```
4.  **Run the Publisher:** Start the `publisher.py` script on the rover (or a second PC for testing).
    ```bash
    python publisher.py
    ```