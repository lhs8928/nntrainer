"""
Device utilities for nntrainer benchmark.
Device interaction including temperature monitoring, cooling, and device info.
"""

import subprocess
import time


def get_thermal_temp():
    """Get thermal zone temperature in Celsius."""
    try:
        cmd = ["adb", "shell", "cat", "/sys/class/thermal/thermal_zone0/temp"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip()) / 1000.0
    except Exception as e:
        print(f"Error reading temp: {e}")
    return 0.0


def wait_for_cooling(target_temp=40.0, max_wait=300, poll_interval=10):
    """
    Wait for device to cool down to target temperature.
    
    Args:
        target_temp: Target temperature in Celsius (default: 40.0)
        max_wait: Maximum wait time in seconds (default: 300 = 5 min)
        poll_interval: Time between temperature checks (default: 10 seconds)
    
    Returns:
        True if target temperature reached, False if timeout
    """
    current_temp = get_thermal_temp()
    print(f"Current temp: {current_temp:.1f}°C, Target: {target_temp:.1f}°C")
    
    if current_temp <= target_temp:
        print(f"Temperature already at target ({current_temp:.1f}°C ≤ {target_temp:.1f}°C)")
        return True
    
    print(f"Cooling down device... (Max wait: {max_wait}s)")
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait:
        time.sleep(poll_interval)
        current_temp = get_thermal_temp()
        print(f"  Current temp: {current_temp:.1f}°C")
        
        if current_temp <= target_temp:
            print(f"Reached target temperature ({current_temp:.1f}°C)")
            return True
    
    print(f"Warning: Timeout waiting for cooling. Current temp: {current_temp:.1f}°C")
    return False


def get_device_model():
    """Get device model name from system properties."""
    try:
        cmd = ["adb", "shell", "getprop", "ro.product.model"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            device = result.stdout.strip()
            return device
    except Exception as e:
        print(f"Error reading device model: {e}")
    return "Unknown"


def get_model_size(model_path, nntr_cfg):
    """Get model file size in human-readable format from device."""
    try:
        # Get the binary file name from config
        model_file = nntr_cfg.get("model_file_name", "model.bin")
        
        # Always get from device (that's where inference runs)
        device_path = f"{model_path}/{model_file}"
        
        # Get file size via adb
        cmd = ["adb", "shell", "wc", "-c", device_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            size_bytes = int(result.stdout.strip().split()[0])
            return format_size(size_bytes)
        else:
            print(f"Warning: Could not get model size from device: {result.stderr}")
    except Exception as e:
        print(f"Error getting model size: {e}")
    return "Unknown"


def format_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KiB', 'MiB', 'GiB']:
        if size_bytes >= 1024.0:
            size_bytes /= 1024.0
        else:
            break
    return f"{size_bytes:.2f} {unit}"
