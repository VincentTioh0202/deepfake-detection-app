import os
import tempfile
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


model = load_model()

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=30,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,
    device=device
)

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_faces(video_path):
    cap = cv2.VideoCapture(video_path)
    faces = []
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

        if len(faces) >= MAX_FRAMES:
            break

        frame_count += 1

    cap.release()
    return faces


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


def predict_video(video_path):
    faces = extract_faces(video_path)

    if len(faces) == 0:
        return None

    frame_probs = [predict_face(face) for face in faces]

    video_prob = float(np.mean(frame_probs))
    prediction = "Fake" if video_prob >= THRESHOLD else "Real"

    if prediction == "Fake":
        selected_idx = int(np.argmax(frame_probs))
    else:
        selected_idx = int(np.argmin(frame_probs))

    selected_face = faces[selected_idx]
    original_img, gradcam_img = generate_gradcam(selected_face)

    return {
        "prediction": prediction,
        "fake_probability": video_prob,
        "threshold": THRESHOLD,
        "frames_used": len(faces),
        "original_img": original_img,
        "gradcam_img": gradcam_img
    }


st.title("Deepfake Detection Using AI")
st.write("Upload a video to classify whether it is Real or Fake with Grad-CAM explainability.")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.video(video_path)

    if st.button("Run Detection"):
        with st.spinner("Analysing video..."):
            result = predict_video(video_path)

        if result is None:
            st.error("No valid face detected in the video.")
        else:
            st.subheader("Prediction Result")
            st.write(f"Prediction: **{result['prediction']}**")
            st.write(f"Fake Probability: **{result['fake_probability']:.4f}**")
            st.write(f"Threshold Used: **{result['threshold']}**")
            st.write(f"Frames Analysed: **{result['frames_used']}**")

            if result["prediction"] == "Fake":
                st.warning("The video is predicted as Fake.")
            else:
                st.success("The video is predicted as Real.")

            st.subheader("Grad-CAM Explainability")

            col1, col2 = st.columns(2)

            with col1:
                st.image(result["original_img"], caption="Selected Face Frame")

            with col2:
                st.image(result["gradcam_img"], caption="Grad-CAM Heatmap")

            st.write(
                "Warmer regions in the heatmap indicate facial areas that contributed more strongly to the model prediction."
            )
