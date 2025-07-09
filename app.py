import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import sys
import cv2
import tempfile
import time
import platform


# =============================
# 🔽 Download model dari Google Drive (jika belum ada)
# =============================
import gdown

if not os.path.exists("model_aspal_vs_nonaspal.h5"):
    st.info("📥 Mengunduh model klasifikasi jalan dari Google Drive...")
    url = "https://drive.google.com/uc?id=11I7KH0_-Hu7uPoWZwpiHQx2u62CHtoE"
    gdown.download(url, "model_aspal_vs_nonaspal.h5", quiet=False)

# =============================
# 🔽 Import dan load model
# =============================

# Load model klasifikasi jalan
from keras.models import load_model
clf_model = load_model("model_aspal_vs_nonaspal.h5")

# Tambah path ke YOLOv5
sys.path.append("yolov5")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.plots import Annotator, colors
from utils.dataloaders import LoadImages
from utils.torch_utils import select_device

# Fungsi scale_coords (manual ambil dari YOLO)
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

# Load YOLOv5
device = select_device('')
model = DetectMultiBackend('best.pt', device=device)

# Preprocessing klasifikasi
def preprocess_tf(img_pil):
    img_resized = img_pil.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(img_pil):
    img_array = preprocess_tf(img_pil)
    pred = clf_model.predict(img_array)[0][0]
    return pred

def detect_damage(img_path):
    dataset = LoadImages(img_path, img_size=640, stride=32, auto=True)
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45)

        for i, det in enumerate(pred):
            annotator = Annotator(im0s.copy(), line_width=3)
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
            result = annotator.result()
    return Image.fromarray(result)

# ===================================
# 🖼️ Upload Gambar
# ===================================
st.title("🖼️ Deteksi Kerusakan Jalan dari Gambar")

uploaded_file = st.file_uploader("Upload gambar jalan", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Diupload", use_container_width=True)

    if st.button("🔍 Deteksi Gambar"):
        score = classify_image(img)
        if score < 0.5:
            st.success("Aspal terdeteksi! Mendeteksi kerusakan...")
            img.save("temp.jpg")
            result_img = detect_damage("temp.jpg")
            st.image(result_img, caption="✅ Hasil Deteksi Kerusakan", use_container_width=True)
        else:
            st.warning("⚠️ Bukan jalan aspal, tidak dilakukan deteksi.")

# ===================================
# 🎥 Webcam Real-time
# ===================================
IS_CLOUD = "streamlit" in sys.executable.lower() or "cloud" in platform.node().lower()

if not IS_CLOUD:
    st.title("📷 Deteksi Real-Time via Webcam")

    run_webcam = st.checkbox("🔴 Nyalakan Kamera")
    frame_window = st.image([])
    frame_count = 0

    if run_webcam:
        camera = cv2.VideoCapture(0)

        while run_webcam:
            success, frame = camera.read()
            if not success:
                st.error("❌ Gagal mengakses webcam.")
                break

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            frame_resized = cv2.resize(frame, (128, 128))
            image_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

            score = classify_image(image_pil)

            if score < 0.5:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                    full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    Image.fromarray(full_rgb).save(tmpfile.name)

                    with torch.no_grad():
                        result_img = detect_damage(tmpfile.name)

                frame_window.image(result_img, caption="✅ Aspal - Deteksi Kerusakan", use_container_width=True)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb, caption="⚠️ Bukan jalan aspal", use_container_width=True)

            time.sleep(0.05)

        camera.release()
        st.write("🟢 Kamera dimatikan.")
else:
    st.title("📷 Deteksi Real-Time via Webcam")
    st.warning("⚠️ Webcam tidak tersedia di Streamlit Cloud.")
