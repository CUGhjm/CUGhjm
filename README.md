## Hi there 👋

<!--
**CUGhjm/CUGhjm** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

基于 RDK X5 的机器视觉导盲头盔 本项目为全国大学生嵌入式芯片与系统设计竞赛 2025 自主命题作品，基于 RDK X5 边缘计算平台设计并实现了一款智能导盲头盔，通过激光雷达、YOLOv5m 视觉检测、IMU 跌倒检测、环境光感知与多模态交互，实现实时障碍物检测、路径引导、夜间自动提示和跌倒检测报警，提升视障用户复杂环境下的安全独立出行能力。主要特性包括：YOLOv5m 剪枝量化实时检测（20FPS）、激光雷达 ±5cm 精度补盲区测距、IMU 动态阈值跌倒检测自动定位报警、环境光感应自动 LED 夜间提示、语音与振动多模态交互、完全边缘计算无需云端依赖。硬件配置：RDK X5（10 TOPS 边缘算力）、N10P 激光雷达（7m）、1080p RGB 摄像头、MPU6050 IMU、环境光传感器、LED 灯、振动马达、蜂鸣器、4G 模块（GPS 定位）。项目结构包含 models（YOLOv5m 模型）、scripts（雷达/YOLO检测/IMU跌倒/环境光检测/交互反馈/main）、dataset（数据集示例）、docs（报告和流程图）、requirements.txt 和 README.md。使用方法：硬件接好后在 Ubuntu 22.04 + Python 3.8+ 下执行 pip install -r requirements.txt 安装依赖，再运行 python scripts/main.py 启动系统，即可实现障碍检测提示、夜间弱光自动开灯、跌倒检测报警与定位、语音振动提示等功能。训练说明：使用 LabelImg 标注自采集街道和障碍场景图像（3500+），使用 YOLOv5 官方库进行训练剪枝量化后导出 .pt 模型部署至 RDK X5，详细见 docs/system_design.pdf。实测性能：检测准确率 92.3%，延迟 ≤100ms，激光雷达测距 7m，跌倒检测响应 2s，功耗 32W，续航 6h。可扩展方向：支持红外摄像头夜视增强、接入智慧城市红绿灯信息、云端同步大数据分析路径优化、适配消防矿山等安全帽场景。欢迎贡献优化检测算法、传感器适配与体验改进，项目基于 MIT License 开源，用于公益科研场景，致力于推动科技助残普惠落地，真正帮助视障群体安全出行。

