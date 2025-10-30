#moving only forward
#!/usr/bin/env python3
"""
Minimal rover movement script for DDSM motor controller.

This script sends simple JSON motor commands over a serial link to a
DDSM-like controller and moves the rover forward for a given distance
using a time-based fallback (duration = distance / speed).

CLI examples (PowerShell on Windows):
  python deliveryrobo.py --port COM3 --distance 1.0 --speed 0.2

Notes:
- This uses a simple percent-based command (0-100). Adjust `max_speed_mps`
  to match your rover physical max speed so percent -> m/s is accurate.
- The JSON command format matches the existing project code that used
  {"T":10010, "id":..., "cmd": <speed_percent>, "act": 3}.
  Verify these fields with your DDSM controller doc; adjust if needed.
- Right motor command is inverted in many gearboxes. Default follows the
  project's original inversion: right motor `cmd` is negated.
"""
import argparse
import json
import serial
import time
import sys


def build_motor_command(motor_id: int, cmd_value: int) -> dict:
    """Build the JSON command expected by the DDSM controller."""
    return {"T": 10010, "id": motor_id, "cmd": int(cmd_value), "act": 3}


def send_command(ser, cmd: dict):
    # kept for backward compatibility but prefer send_command_with_options below
    payload = json.dumps(cmd) + "\n"
    ser.write(payload.encode("utf-8"))


def stop_motors(ser, left_id: int, right_id: int, repeats: int = 3, delay: float = 0.05):
    cmd_left = build_motor_command(left_id, 0)
    cmd_right = build_motor_command(right_id, 0)
    for _ in range(repeats):
        send_command(ser, cmd_right)
        time.sleep(delay)
        send_command(ser, cmd_left)
        time.sleep(delay)


def send_command_with_options(ser, cmd: dict, terminator: str = "\n", read_response: bool = True, resp_timeout: float = 0.2):
    """Send JSON command with configurable terminator and optionally read a short response.

    Returns the raw bytes read (may be empty bytes).
    """
    payload = json.dumps(cmd) + terminator
    ser.write(payload.encode("utf-8"))
    ser.flush()
    if not read_response:
        return b""
    # read available response for a short time
    end = time.time() + resp_timeout
    buf = bytearray()
    try:
        while time.time() < end:
            n = ser.in_waiting if hasattr(ser, 'in_waiting') else 0
            if n:
                buf.extend(ser.read(n))
            else:
                # small sleep to wait for data
                time.sleep(0.01)
    except Exception:
        pass
    return bytes(buf)


def main():
    parser = argparse.ArgumentParser(description="Move rover a short distance via DDSM controller over serial.")
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM3 or /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--distance", type=float, default=1.0, help="Distance to travel in meters (default: 1.0)")
    parser.add_argument("--speed", type=float, default=0.2, help="Forward speed in m/s (default: 0.2)")
    parser.add_argument("--max-speed", type=float, default=0.6, help="Rover top speed in m/s that maps to 100%% (default: 0.6)")
    parser.add_argument("--left-id", type=int, default=1, help="Left motor id on controller (default: 1)")
    parser.add_argument("--right-id", type=int, default=2, help="Right motor id on controller (default: 2)")
    parser.add_argument("--freq", type=float, default=10.0, help="Command send frequency in Hz (default: 10)")
    args = parser.parse_args()

    if args.speed <= 0:
        print("Error: --speed must be > 0")
        sys.exit(1)

    # Estimate runtime based on distance and speed
    duration = args.distance / args.speed
    print(f"Moving {args.distance:.2f} m at {args.speed:.2f} m/s -> duration {duration:.2f} s")

    # Map speed (m/s) to percent (0-100)
    speed_percent = int(round((args.speed / args.max_speed) * 100.0))
    speed_percent = max(0, min(100, speed_percent))
    print(f"Using speed percent: {speed_percent}% (max_speed_mps={args.max_speed})")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except Exception as e:
        print(f"Failed to open serial port {args.port}: {e}")
        sys.exit(1)

    interval = 1.0 / args.freq
    end_time = time.time() + duration

    try:
        print("Starting movement loop — press Ctrl-C to abort")
        while time.time() < end_time:
            # build commands. Note: right motor value is negated following project convention
            cmd_right = build_motor_command(args.right_id, -speed_percent)
            cmd_left = build_motor_command(args.left_id, speed_percent)

            send_command(ser, cmd_right)
            # short gap to prevent overload
            time.sleep(0.01)
            send_command(ser, cmd_left)

            time.sleep(interval - 0.01 if interval > 0.01 else interval)

        print("Target distance reached -> stopping motors")
        stop_motors(ser, args.left_id, args.right_id)
        print("Done")

    except KeyboardInterrupt:
        print("Interrupted -> stopping motors")
        stop_motors(ser, args.left_id, args.right_id)

    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
