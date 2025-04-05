# Traffic-Violation-Detection-System
# 🚗 AI-Powered Traffic Violation Detection System

This project is a computer vision-based system that detects and logs traffic violations from video footage. It uses a pre-trained AI model to identify vehicles and analyze behaviors such as red-light violations, wrong-side driving, and potential crashes.

The goal is to demonstrate how artificial intelligence and computer vision can be used to improve road safety and automate traffic monitoring.

---

## 🚀 Features

- ✅ Detects vehicles using YOLOv8
- 🔴 Red-light violation detection
- 🔄 Wrong-side driving detection
- 💥 Basic crash detection (proximity-based)
- 📸 Stores screenshots of violations
- 📝 Logs violations for analysis

---

## 💡 How It Works

1. The system processes a video **frame by frame**
2. It uses YOLOv8 (trained on COCO dataset) to detect objects
3. Logic is applied to identify specific traffic violations
4. Frames with violations are saved and optionally logged

---

## 🛠️ Tech Stack

| Tool/Library     | Purpose                            |
|------------------|-------------------------------------|
| **Python**       | Main programming language           |
| **YOLOv8 (Ultralytics)** | Object detection model trained on COCO |
| **OpenCV**       | Frame processing and image handling |
| **Google Colab** | Development and execution environment |
| **NumPy**        | Numerical operations (distance, tracking) |

The Backend has not been connected 



