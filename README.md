## Hi there 👋

<!--
**CUGhjm/CUGhjm** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

# Smart Blind Guide Helmet (RDK X5)

A lightweight, real-time smart guide helmet for visually impaired users, enabling **safe and independent outdoor travel**. Built on the RDK X5 edge AI platform, it combines LiDAR, YOLOv5m detection, IMU-based fall detection, ambient light sensing, and multimodal feedback to actively detect obstacles, recognize traffic signs, and provide real-time voice and vibration alerts.

**Features:**
- Real-time object detection (YOLOv5m, pruned & quantized, 20 FPS)
- LiDAR ±5cm accurate distance measurement up to 7m
- IMU-based dynamic fall detection with auto GPS alert
- Auto LED for low-light conditions
- Voice and vibration feedback
- Fully edge computing, no cloud dependency

**Hardware:**
RDK X5 (10 TOPS NPU), N10P LiDAR, 1080p camera, MPU6050 IMU, ambient light sensor, LED, vibration motor, buzzer, 4G GPS module.

**Project Structure:**
models/ (YOLOv5m models), scripts/ (LiDAR, YOLO, IMU, LED, feedback, main), dataset/ (sample data), docs/ (PDF, diagrams), requirements.txt, README.md.

**Quick Start:**
- Ubuntu 22.04 + Python 3.8+
- Connect hardware (LiDAR, camera, IMU, LED, GPS)
- Install dependencies: `pip install -r requirements.txt`
- Run: `python scripts/main.py`
- Features: obstacle detection, voice/vibration alerts, auto LED in low-light, fall detection with GPS alerts.

**Training:**
- Annotate 3500+ images with LabelImg
- Train YOLOv5m with pruning & quantization
- Deploy `.pt` model on RDK X5
- Detailed guide in `docs/system_design.pdf`.

**Performance:**
- Detection Accuracy: 92.3%
- Latency: ≤100ms
- LiDAR Range: 7m
- Fall Response: 2s
- Power: 32W
- Runtime: 6h

**Future Work:**
Infrared night vision, smart traffic light integration, cloud-based route analysis, and adaptation for safety helmets in special industries.

---

**License:** MIT

If this project helps you, consider ⭐ starring it to support future development.




