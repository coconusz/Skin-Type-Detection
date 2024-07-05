import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install required packages
try:
    import cv2
except ImportError:
    install('opencv-python-headless')
    import cv2

try:
    import torch
except ImportError:
    install('torch')
    import torch

try:
    import torchvision
except ImportError:
    install('torchvision')
    import torchvision

try:
    import facenet_pytorch
except ImportError:
    install('facenet-pytorch')
    import facenet_pytorch

try:
    import streamlit_webrtc
except ImportError:
    install('streamlit-webrtc')
    import streamlit_webrtc

import streamlit as st
import torch.nn as nn
from facenet_pytorch import MTCNN
from torchvision import transforms, models
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import av

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Define the CustomResNet model
class CustomResNet(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer2[1].conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer3[1].conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer4[1].conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load the pre-trained model
model = CustomResNet(num_classes=4)
try:
    model.load_state_dict(torch.load('Dashboard/skintypes-model.pth', map_location=torch.device('cpu')), strict=False)
    model.eval()
except RuntimeError as e:
    st.error(f"Error loading the model: {e}")

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Skin types and their descriptions
skin_types = ["Berminyak", "Kering", "Normal", "Kombinasi"]
skin_type_descriptions = {
    "Berminyak": "Kulitmu terdeteksi berminyak. Tipe kulit yang berminyak cenderung terlihat mengkilap dan licin akibat produksi minyak atau sebum berlebih pada wajah.",
    "Kering": "Kulitmu terdeteksi kering. Tipe kulit kering memiliki tingkat kelembapan yang rendah. Secara umum, orang yang memiliki tipe kulit kering kerap kali menghadapi masalah kulit, yakni mudah iritasi, sehingga rentan mengalami kemerahan dan jerawat.",
    "Normal": "Kulitmu terdeteksi normal. Seseorang yang memiliki kulit normal, tingkat sebum atau minyaknya dan tingkat hidrasi pada kulitnya seimbang, sehingga kulit tipe ini tidak terlalu kering dan tidak berminyak.",
    "Kombinasi": "Kulitmu terdeteksi kombinasi. Jenis kulit kombinasi merupakan perpaduan antara kulit berminyak dengan kulit kering. Seseorang dengan jenis kulit kombinasi memiliki kulit berminyak di area T-zone, yakni area dahu, hidung, dan dagu, serta kulit kering di area pipi."
}
skin_type_care = {
    "Berminyak": "Saran Perawatan: Menggunakan pembersih wajah yang diformulasikan khusus untuk kulit berminyak, yang biasanya mengandung gliserin. Setelah selesai mencuci wajah, kamu bisa menggunakan produk perawatan kulit lain, seperti toner yang mengandung asam salisilat, benzoil peroksida, dan asam glikolat.",
    "Kering": "Saran Perawatan: Tipe kulit kering membutuhkan hidrasi lebih. Jadi, kamu perlu menggunakan produk yang mampu melembapkan kulit wajah, yang biasanya mengandung emolien. Tipe kulit kering harus menghindari produk perawatan dan kosmetik yang mengandung alkohol dan pewangi.",
    "Normal": "Saran Perawatan: Cukup gunakan sabun cuci wajah yang gentle dan hindari menggosok wajah dengan kasar. Kamu juga bisa menggunakan air hangat untuk membasuh wajah, kemudian mengeringkannya dengan tisu atau handuk bersih berbahan lembut.",
    "Kombinasi": "Saran Perawatan: Bersihkan wajah 2 kali sehari secara rutin. Hindari penggunaan produk pembersih yang mengandung alkohol, asam salisilat, dan benzoil peroksida. Selain itu, kamu juga bisa menggunakan produk pembersih untuk kulit wajah kering di area pipi dan produk khusus kulit berminyak untuk area T-zone."
}

# Preprocess the image
def preprocess_image(image):
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Classify the image using the model
def classify_image(image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_skin_type = skin_types[predicted.item()]
    return predicted_skin_type

# Detect faces in the image
def detect_faces(image):
    boxes, _ = mtcnn.detect(image)
    return boxes

# Streamlit app
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("✨Skin Type Detection✨")
st.write("Kenali Tipe Wajahmu dengan Kamera!")

# VideoTransformer for webcam input
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.result_img = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.result_img = img
        return img

def take_photo():
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
    )
    st.write("Klik tombol START kemudian posisikan wajahmu pada kamera dan klik tombol di bawah untuk mengambil gambar.")
    
    if st.button('📸 Ambil Foto'):
        if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.result_img is not None:
            captured_img = webrtc_ctx.video_transformer.result_img
            st.image(captured_img, use_column_width=True)
            st.write("")
            st.write("Processing...")

            captured_img_rgb = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
            face_image_pil = Image.fromarray(captured_img_rgb)
        
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
            if len(faces) == 0:
                st.write("Wajah tidak terdeteksi! Silahkan ambil gambar kembali.")
            else:
                x, y, w, h = faces[0]
                face_image = captured_img[y:y+h, x:x+w]
                face_image_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

                processed_image = preprocess_image(face_image_pil)
                predicted_skin_type = classify_image(processed_image)

                st.write(f"Jenis Kulit yang Terdeteksi: {predicted_skin_type}")
                st.write(skin_type_descriptions[predicted_skin_type])
                st.write(skin_type_care[predicted_skin_type])
        else:
            st.write("Gambar belum diambil atau tidak ditemukan.")

take_photo()
