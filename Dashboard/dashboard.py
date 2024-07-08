import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np

mtcnn = MTCNN()

skin_types = ["Berminyak", "Kering", "Normal", "Kombinasi"]

def load_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(skin_types))  

    state_dict = torch.load('C:\\Documents\\COLLEGE\\PI\\Implementasi Algoritma Convolutional Neural Network (CNN) Untuk Sistem Identifikasi Jenis Kulit Wajah\\Dashboard\\skintypes-model.pth', map_location=torch.device('cpu'))
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
    model.load_state_dict(state_dict, strict=False)

    model.fc.weight.data = torch.nn.init.xavier_uniform_(model.fc.weight.data)
    model.fc.bias.data.fill_(0)

    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    image = Image.fromarray(image)

    image = transform(image)
    image = image.unsqueeze(0)
    return image

def classify_image(image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_skin_type = skin_types[predicted.item()]
    return predicted_skin_type

st.markdown("""
    <style>
        .title {
            font-size: 55px;
            color: #ffffff;
            text-align: center;
        }
        .description {
            font-size: 15px;
            color: #B4AFAE;
            text-align: center;
        }
        .result {
            font-size: 17px;
            color: #ffffff;
        }
        .care {
            font-size: 17px;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown('<h1 class="title">✨Skin Type Detection✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Klik tombol dibawah untuk mengambil gambar dan sistem akan mendeteksi jenis kulit wajahmu!</p>', unsafe_allow_html=True)

def take_photo():
    video_stream = st.camera_input("")
    if video_stream is not None:
        frame = np.array(Image.open(video_stream))

        boxes, _ = mtcnn.detect(frame)
        
        if boxes is not None:
            for box in boxes:
                x, y, w, h = [int(coord) for coord in box]
                face_image = frame[y:y+h, x:x+w]
                
                face_image_np = np.array(face_image)

                processed_image = preprocess_image(face_image_np)
                predicted_skin_type = classify_image(processed_image)

                st.image(face_image, caption=f"Jenis Kulit Wajah: {predicted_skin_type}", use_column_width=True)
                st.markdown(f'<p class="result">{skin_type_descriptions[predicted_skin_type]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="care">{skin_type_care[predicted_skin_type]}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="description">Wajah tidak terdeteksi. Silahkan ambil gambar lagi.</p>', unsafe_allow_html=True)

take_photo()
