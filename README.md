# AquaSight.

An intelligent, cinematic vision system designed for high-end marine analysis. AquaSight replaces standard boilerplate AI dashboards with a bespoke, "Creative Studio" aesthetic inspired by modern minimalist web design—bringing a highly polished, interactive experience to complex neural network inference.

## ✨ Features

- **Cinematic Inference Engine**: Real-time YOLOv8 bounding box plotting with smooth, dynamic transitions and native heatmap integration.
- **Glass-Refraction UI**: A complete frontend overhaul of Streamlit using responsive "Bento Grid" cards, custom spring-physics cursors, and a global Light/Dark Liquid Theme engine.
- **Classification Matrix Export**: Instantly parse the active detection frame into a tabular dataset and export the raw confidence data as a CSV matrix.
- **Dynamic Neural Tracking**: Live system health telemetry, including real-time latency (MS) rendering and dynamic Neural Load tracking.
- **Integrated Model Architecture**: View comprehensive training curves, F1-Confidence metrics, and Normalized Confusion Matrices directly within the app's routing framework.

## 🧠 The Model & Dataset

AquaSight is powered by **YOLOv8** and trained on the **Fish Detection v5** dataset sourced from Roboflow Universe.

- **Dataset Scope**: 8,242 meticulously annotated high-resolution images.
- **Taxonomy**: Capable of detecting 13 distinct marine species (e.g., AngelFish, ClownFish, BlueTang, ZebraFish, MorishIdol, etc.).
- **Augmentations applied**: Gaussian blurring, exposure adjustments, and rotational offsets to ensure robust detection regardless of water turbidity.

**Evaluation Profile (10 Epochs):**
- **mAP50**: 89.8%
- **Precision**: 83.7%
- **Recall**: 86.7%

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RyanV-0407/AquaSight.git
   cd AquaSight
   ```

2. **Install the dependencies:**
   Make sure you have Python installed. Then, run:
   ```bash
   pip install streamlit ultralytics pandas numpy Pillow
   ```

3. **Run the Vision System:**
   ```bash
   streamlit run app.py
   ```

## 👨‍💻 Author
Developed by **Vikram Singh**. Passionate about leveraging computer vision and neural networks for real-world environmental and analytical applications.

*© 2026 AquaSight Studio.*
