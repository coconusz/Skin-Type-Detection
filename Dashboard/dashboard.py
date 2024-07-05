import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from torchvision import transforms, models
from PIL import Image

mtcnn = MTCNN()

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

model = CustomResNet(num_classes=4)
try:
    model.load_state_dict(torch.load('Dashboard/skintypes-model.pth', map_location=torch.device('cpu')), strict=False)
    model.eval()
except RuntimeError as e:
    st.error(f"Error loading the model: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

def preprocess_image(image):
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def classify_image(image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_skin_type = skin_types[predicted.item()]
    return predicted_skin_type

def detect_faces(image):
    boxes, _ = mtcnn.detect(image)
    return boxes

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("✨Skin Type Detection✨")
st.write("Kenali Tipe Wajahmu dengan Kamera!")

choice = st.sidebar.selectbox("Pilih Mode Input", ["Gambar", "Kamera"])

if choice == "Kamera":
    st.markdown('<p class="header-font">Kamera</p>', unsafe_allow_html=True)
    st.markdown('<p class="description-font">Hanya dapat diakses atau digunakan dengan kamera webcam (desktop).</p>', unsafe_allow_html=True)
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        # To read image file buffer with PIL:
        img_pil = Image.open(img_file_buffer)
        
        # Process the image
        img_rgb = np.array(img_pil)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            st.write("Wajah tidak terdeteksi! Silahkan ambil gambar kembali.")
        else:
            x, y, w, h = faces[0]
            face_image = img_rgb[y:y+h, x:x+w]
            face_image_pil = Image.fromarray(face_image)
            
            processed_image = preprocess_image(face_image_pil)
            prediction = classify_image(processed_image)
            
            # Display the prediction
            st.write(f"Jenis Kulit yang Terdeteksi: {prediction}")
            st.write(skin_type_descriptions[prediction])
            st.write(skin_type_care[prediction])
            
            # Display the image with the prediction text
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_bgr, prediction, (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            st.image(img_bgr, channels="BGR")

elif choice == "Gambar":
    st.markdown('<p class="header-font">Gambar</p>', unsafe_allow_html=True)
    st.markdown('<p class="description-font">Unggah gambar dari komputer.</p>', unsafe_allow_html=True)
    img_file_buffer = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    if img_file_buffer is not None:
        # To read image file buffer with PIL:
        img_pil = Image.open(img_file_buffer)
        
        # Process the image
        img_rgb = np.array(img_pil)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            st.write("Wajah tidak terdeteksi! Silahkan unggah gambar lain.")
        else:
            x, y, w, h = faces[0]
            face_image = img_rgb[y:y+h, x:x+w]
            face_image_pil = Image.fromarray(face_image)
            
            processed_image = preprocess_image(face_image_pil)
            prediction = classify_image(processed_image)
            
            # Display the prediction
            st.write(f"Jenis Kulit yang Terdeteksi: {prediction}")
            st.write(skin_type_descriptions[prediction])
            st.write(skin_type_care[prediction])
            
            # Display the image with the prediction text
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_bgr, prediction, (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            st.image(img_bgr, channels="BGR")
