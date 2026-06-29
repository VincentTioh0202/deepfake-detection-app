import os
import tempfile
import subprocess
import gdown
import cv2
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from facenet_pytorch import MTCNN

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget


IMAGE_SIZE = 384
THRESHOLD = 0.51
MAX_FRAMES = 25
FRAME_INTERVAL = 8
MODEL_PATH = "best_model.pth"
GOOGLE_DRIVE_FILE_ID = "1EAt0A7qYK7Gbuvzksu3hw5zLGVeQQ94F"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EfficientNetV2Deepfake(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        self.backbone = models.efficientnet_v2_m(weights=None)
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)


@st.cache_resource
def load_model():
    download_model()
    model = EfficientNetV2Deepfake()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_mtcnn():
    return MTCNN(
        image_size=IMAGE_SIZE,
        margin=30,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        device=device
    )


model = load_model()
mtcnn = load_mtcnn()


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def convert_preview_to_h264(input_path):
    preview_path = input_path + "_preview_h264.mp4"

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "aac",
        "-movflags", "+faststart",
        preview_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        st.error("FFmpeg conversion failed:")
        st.code(result.stderr)
        return None

    return preview_path


def extract_faces(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return [], []

    faces = []
    frame_indices = []
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            boxes, probs = mtcnn.detect(pil_img)

            if boxes is not None and probs is not None:
                best_idx = int(np.argmax(probs))

                if probs[best_idx] >= 0.90:
                    x1, y1, x2, y2 = map(int, boxes[best_idx])
                    h, w, _ = frame_rgb.shape

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        face = frame_rgb[y1:y2, x1:x2]
                        face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
                        faces.append(Image.fromarray(face))
                        frame_indices.append(frame_count)

        if len(faces) >= MAX_FRAMES:
            break

        frame_count += 1

    cap.release()
    return faces, frame_indices


def predict_face(face):
    img = eval_transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        prob_fake = torch.sigmoid(output).item()

    return prob_fake


def generate_gradcam(face):
    input_tensor = eval_transform(face).unsqueeze(0).to(device)
    target_layers = [model.backbone.features[-2]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[BinaryClassifierOutputTarget(1)],
            aug_smooth=True,
            eigen_smooth=True
        )[0]

    rgb_img = np.array(face.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0

    heatmap = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return rgb_img, heatmap


def evaluate_video_level(video_path):
    faces, frame_indices = extract_faces(video_path)

    if len(faces) == 0:
        return None

    frame_probs = []

    for face in faces:
        prob = predict_face(face)
        frame_probs.append(prob)

    frame_probs = np.array(frame_probs)

    video_fake_probability = float(np.mean(frame_probs))
    video_prediction = "Fake" if video_fake_probability >= THRESHOLD else "Real"

    if video_prediction == "Fake":
        selected_idx = int(np.argmax(frame_probs))
    else:
        selected_idx = int(np.argmin(frame_probs))

    selected_face = faces[selected_idx]
    original_img, gradcam_img = generate_gradcam(selected_face)

    frame_result_df = {
        "Frame Index": frame_indices,
        "Fake Probability": frame_probs.round(4),
        "Frame Prediction": [
            "Fake" if p >= THRESHOLD else "Real"
            for p in frame_probs
        ]
    }

    return {
        "video_prediction": video_prediction,
        "video_fake_probability": video_fake_probability,
        "threshold": THRESHOLD,
        "frames_used": len(faces),
        "average_method": "Mean aggregation of frame-level fake probabilities",
        "frame_result_df": frame_result_df,
        "selected_frame_index": frame_indices[selected_idx],
        "selected_frame_probability": float(frame_probs[selected_idx]),
        "original_img": original_img,
        "gradcam_img": gradcam_img
    }


st.title("Deepfake Detection Using AI")

st.write(
    "Upload a video to classify whether it is Real or Fake. "
    "The system uses the original uploaded video for detection, while a converted copy is used only for preview."
)

uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    video_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(video_bytes)
        video_path = temp_file.name

    st.subheader("Uploaded Video Preview")

    with st.spinner("Preparing video preview..."):
        preview_path = convert_preview_to_h264(video_path)

    if preview_path is not None:
        st.video(preview_path)
    else:
        st.warning(
            "Video preview is unavailable because FFmpeg conversion failed. "
            "Detection can still run using the original video."
        )

    st.caption("Detection uses the original uploaded video. The H.264 copy is only for browser preview.")

    if st.button("Run Detection"):
        with st.spinner("Analysing original video..."):
            result = evaluate_video_level(video_path)

        if result is None:
            st.error("No valid face detected in the video, or the video cannot be read.")
        else:
            st.subheader("Video-Level Prediction Result")

            st.write(f"Final Prediction: **{result['video_prediction']}**")
            st.write(f"Video Fake Probability: **{result['video_fake_probability']:.4f}**")
            st.write(f"Threshold Used: **{result['threshold']}**")
            st.write(f"Frames Analysed: **{result['frames_used']}**")
            st.write(f"Aggregation Method: **{result['average_method']}**")

            if result["video_prediction"] == "Fake":
                st.warning("The uploaded video is predicted as Fake.")
            else:
                st.success("The uploaded video is predicted as Real.")

            st.subheader("Frame-Level Prediction Summary")
            st.dataframe(result["frame_result_df"])

            st.subheader("Grad-CAM Explainability")

            st.write(
                f"Grad-CAM is generated from frame index "
                f"**{result['selected_frame_index']}**, with fake probability "
                f"**{result['selected_frame_probability']:.4f}**."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.image(result["original_img"], caption="Selected Face Frame")

            with col2:
                st.image(result["gradcam_img"], caption="Grad-CAM Heatmap")

            st.write(
                "Warmer regions in the heatmap indicate facial areas that contributed more strongly "
                "to the model prediction."
            )
