# 🛣️ Road Damage Detection using CNN and YOLOv5

This project aims to build an automated road damage detection system using **deep learning**. The system consists of two main stages: **road type classification (asphalt or non-asphalt)** using CNN and **damage type detection** such as cracks or potholes using YOLOv5. The system is implemented as a web application using **Streamlit**.

---

## 🎯 Objectives  
- Detect whether an image contains **asphalt road or not**.
- If it is an asphalt road, detect the **type of road damage** present in the image.
- Provide a **web-based interface** that works with both **image uploads** and **real-time webcam**.

---

## 💡 Benefits  
- **Reduce reliance on manual inspections.**  
- **Increase speed and accuracy** in road damage detection.  
- **Assist government agencies** in infrastructure maintenance.  
- **Support the development of smart cities.**

---

## 📏 Scope Limitations  
- The system only detects **damage on asphalt roads**.  
- Detection works with images taken from a top-down or straight-ahead perspective.  
- Models are trained only on specific datasets (RTK & Road Damage Dataset).

---

## 🧠 Model Architecture
Image → CNN (Asphalt / Non-Asphalt) →
If Asphalt → YOLOv5 → Bounding Box + Damage Labels

1. **CNN** is used for road type classification.  
2. **YOLOv5** is used for detecting **types of road damage**.

---

## 🧾 Damage Categories (YOLOv5)

The YOLOv5 model detects the following 4 types of road damage:

| Label | Category Code | Description |
|-------|---------------|-------------|
| **D00** | Longitudinal Cracks | Cracks that run in the same direction as the road. |
| **D10** | Transverse Cracks | Cracks that run across the road. |
| **D20** | Alligator Cracks | Interconnected cracks forming a pattern similar to an alligator's skin. |
| **D40** | Potholes | Deep depressions or holes in the road surface. |

These categories follow the labeling format used in the Road Damage Dataset.

---

## 📂 Dataset

| Dataset | Description | Link |
|--------|-------------|------|
| RTK Dataset | For classifying asphalt vs non-asphalt | [Kaggle - RTK](https://www.kaggle.com/datasets/tallwinkingstan/road-traversing-knowledge-rtk-dataset) |
| Road Damage Dataset | For detecting road surface damage | [Kaggle - Road Damage](https://www.kaggle.com/datasets/alvarobasily/road-damage) |

---

## 🧪 Model Training

### 1. CNN for Asphalt / Non-Asphalt Classification  
📁 `Projek_CV_Virlip_(aspal_or_tanah).ipynb`

- CNN architecture using Conv2D and Dense layers.
- Input images resized to 128x128.
- Output: probability of being asphalt (0: asphalt, 1: non-asphalt).
- Trained model saved as `.h5` file (`model_aspal_vs_nonaspal.h5`).

### 2. YOLOv5 for Road Damage Detection  
📁 `Projek_CV_Virlip_(jenis_kerusakan_aspal).ipynb`

- Preprocessing Road Damage Dataset into YOLO format.
- Trained using YOLOv5 from the official Ultralytics repo.
- Output: bounding boxes and damage labels.
- Trained model saved as `best.pt`.

---

## ⚙️ Web Application

### 📁 `app.py`  
Framework: [Streamlit](https://streamlit.io)

### Features:
- Upload image and perform automatic detection.
- Real-time damage detection using webcam.
- Display bounding boxes and damage labels on result images.

### To Run the App:
```bash
streamlit run app.py
