#!/usr/bin/env python3
import subprocess
import os
from threading import Thread

# Predefined audio file mappings
AUDIO_PRESETS = {
    "1": "/app/pydev_demo/12_yolov5s_v6_v7_sample/SOUND/speech1.wav",
    "2": "/app/pydev_demo/12_yolov5s_v6_v7_sample/SOUND/speech2.wav",
    "3": "/app/pydev_demo/12_yolov5s_v6_v7_sample/SOUND/speech4.wav",
    "4": "/app/pydev_demo/12_yolov5s_v6_v7_sample/SOUND/speech3.wav",
    "5": "/app/pydev_demo/12_yolov5s_v6_v7_sample/SOUND/speech5.wav",
    "6": "/app/pydev_demo/12_yolov5s_v6_v7_sample/SOUND/speech6.wav",
    "7": "/app/pydev_demo/12_yolov5s_v6_v7_sample/SOUND/speech7.wav",
    "8": "/app/pydev_demo/12_yolov5s_v6_v7_sample/SOUND/speech8.wav",
    "9": "/app/pydev_demo/12_yolov5s_v6_v7_sample/SOUND/speech9.wav",
    "10": "/app/pydev_demo/12_yolov5s_v6_v7_sample/SOUND/speech10.wav",
    "11": "/app/pydev_demo/12_yolov5s_v6_v7_sample/SOUND/speech11.wav",
}

# ??????????????
current_play_process = None


def play_audio(file_path, device="plughw:0,0"):
    """??????????"""
    global current_play_process

    try:
        if not os.path.exists(file_path):
            print(f"Error: Audio file {file_path} does not exist")
            return False

        # ?????????????????
        if current_play_process and current_play_process.poll() is None:
            current_play_process.terminate()
            current_play_process.wait()

        # ??Popen????
        current_play_process = subprocess.Popen(
            ["sudo", "aplay", "-D", device, file_path],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        print(f"Playing: {file_path}")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def play_by_number(choice):
    """????????"""
    if choice in AUDIO_PRESETS:
        # ???????????????
        def play_thread():
            play_audio(AUDIO_PRESETS[choice])

        thread = Thread(target=play_thread)
        thread.daemon = True
        thread.start()
        return True

    print(f"Wrong:No use{choice}")
    return False


def show_menu():
    """Display the available options"""
    print("\nAudio Player Menu:")
    print("1. Play record1.wav")
    print("2. Play record2.wav")
    print("3. Play record3.wav")
    print("q. Quit")


def interactive_mode():
    print("Simple Audio Player")
    print("------------------")

    while True:
        show_menu()
        choice = input("Enter your choice (1_25/q): ").strip().lower()

        if choice == 'q':
            print("Exiting...")
            break

        play_by_number(choice)


if __name__ == "__main__":
    interactive_mode()
