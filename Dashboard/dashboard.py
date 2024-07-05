import cv2
import streamlit as st
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from torchvision import transforms, models
from PIL import Image

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
model.load_state_dict(torch.load('Dashboard/skintypes-model.pth', map_location=torch.device('cpu')))
model.eval()

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

# Streamlit app
st.title("✨Skin Type Detection✨")
st.write("Kenali Tipe Wajahmu!")

# Function to process camera input
def process_camera_input():
    video_stream = st.camera_input()
    while video_stream.is_active:
        frame = video_stream.read()
        if frame is not None:
            # Convert frame to PIL Image
            frame_pil = Image.fromarray(frame)

            # Detect faces
            boxes, _ = mtcnn.detect(frame_pil)

            # Display only the face if detected
            if boxes is not None:
                for box in boxes:
                    x, y, w, h = [int(coord) for coord in box]
                    face_image = frame[y:y+h, x:x+w]
                    face_image_pil = Image.fromarray(face_image)
                    
                    # Preprocess and classify the face image
                    processed_image = preprocess_image(face_image_pil)
                    predicted_skin_type = classify_image(processed_image)

                    # Display results
                    st.image(face_image, caption=f"Jenis Kulit: {predicted_skin_type}", use_column_width=True)
                    st.write(skin_type_descriptions[predicted_skin_type])
                    st.write(skin_type_care[predicted_skin_type])  

process_camera_input()
