# 👣 AI-Gait-Visualizer

[![Launch on Hugging Face](https://img.shields.io/badge/Launch-🟧%20Hugging%20Face%20Demo-orange)](https://huggingface.co/spaces/rmehmood786/AI-Gait-Visualizer)

---

### 📍 Overview
**AI-Gait-Visualizer** is an interactive application that extracts human pose keypoints and gait metrics from short walking videos.  
It uses **YOLOv8-Pose** for keypoint detection and a lightweight gait-analysis pipeline to estimate stride length proxies and cadence.
---

### 🚀 Live Demo
🟢 **Try it directly here:**  
👉 [https://huggingface.co/spaces/rmehmood786/AI-Gait-Visualizer](https://huggingface.co/spaces/rmehmood786/AI-Gait-Visualizer)

Upload a walking video → click **Run Analysis** → view the annotated video and download CSV metrics.

---

### ⚙️ Features
- 🎯 **Pose Estimation:** Real-time body keypoints using YOLOv8-Pose.  
- 🦶 **Gait Metrics:** Calculates stride proxy, step duration, and approximate cadence.  
- 📊 **Downloadable Results:** Exports a CSV file with per-frame gait data.  
- 💡 **Interactive Interface:** Built with Gradio for smooth use.  
- ☁️ **Cloud-Ready:** Runs directly on Hugging Face Spaces.

---

### 🧩 Tech Stack
| Component | Description |
|------------|-------------|
| **Model** | YOLOv8-Pose (Ultralytics) |
| **Frameworks** | Gradio, OpenCV, NumPy, SciPy, Pandas |
| **Deployment** | Hugging Face Spaces |
| **Language** | Python 3.10+ |

---

### 🗂️ Project Structure
```
AI-Gait-Visualizer/
│
├── app/
│   └── app.py              # Gradio app entry point
│
├── src/
│   └── gait_viz.py         # Gait extraction + pose processing
│
├── requirements.txt         # Dependencies
├── app_file                 # tells Hugging Face to run app/app.py
└── README.md
```

---

### 🧠 How It Works
1. The uploaded video is read frame-by-frame using OpenCV.  
2. YOLOv8-Pose detects body keypoints per frame.  
3. Gait parameters (stride proxy, step duration, cadence) are computed from keypoint motion.  
4. Processed frames are written back into an annotated video.  
5. The app displays results and offers a downloadable CSV.

---

### 🖥️ Run Locally
```bash
git clone https://github.com/rmehmood786/AI-Gait-Visualizer.git
cd AI-Gait-Visualizer
pip install -r requirements.txt
python app/app.py
```
Then open the local Gradio link in your browser.

---

### 📦 Requirements
```
gradio>=4.0.0
ultralytics>=8.2.0
opencv-python>=4.10.0
numpy>=1.24
pandas>=2.1
scipy>=1.11
matplotlib>=3.8
tqdm
rich
```

---

### 🎬 Example Output
| Input | Output |
|-------|---------|
| <img src="https://raw.githubusercontent.com/rmehmood786/AI-Gait-Visualizer/main/Sample_input.gif" width="300"/> | <img src= "sample_output.gif" width="300"/> |
---
## 💡 Ideas to Extend or Experiment With

- Integrate pose estimation with gait recognition for identity prediction.  
- Visualize motion trajectories in 3D using Open3D or Matplotlib.  
- Add real-time gait visualization using webcam input.  
- Try comparing different YOLO variants (v8, v10) for accuracy vs. speed.  
- Use temporal smoothing or Kalman filtering for stable keypoints.  
- Build a web-based dashboard for visualizing stride metrics interactively.
---

### 👤 Author
**Rashid Mehmood**  
AI Researcher — Computer Vision & Gait Analysis  
- 🌐 [GitHub](https://github.com/rmehmood786)  
- 🤗 [Hugging Face Space](https://huggingface.co/spaces/rmehmood786/AI-Gait-Visualizer)

---

### 🪪 License
Released under the **MIT License**.  
You are free to use, modify, and share with proper attribution.

---

### ⭐ Acknowledgements
Built using:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- [Gradio](https://github.com/gradio-app/gradio)