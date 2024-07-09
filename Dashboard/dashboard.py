import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np

# Initialize the face detector
mtcnn = MTCNN()

# Define skin types and descriptions
skin_types = ["Berminyak", "Kering", "Normal", "Kombinasi"]

def load_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(skin_types))  

    # Load the state dictionary, ignoring the fc layer
    state_dict = torch.load('C:\\Documents\\COLLEGE\\PI\\Implementasi Algoritma Convolutional Neural Network (CNN) Untuk Sistem Identifikasi Jenis Kulit Wajah\\Dashboard\\skintypes-model.pth', map_location=torch.device('cpu'))
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
    model.load_state_dict(state_dict, strict=False)

    # Manually initialize the fc layer weights
    model.fc.weight.data = torch.nn.init.xavier_uniform_(model.fc.weight.data)
    model.fc.bias.data.fill_(0)

    model.eval()
    return model

model = load_model()

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Definisi jenis-jenis kulit
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

skin_type_info = {
    "Berminyak": "Kulit berminyak disebabkan oleh produksi sebum yang tinggi. Sebum adalah minyak alami yang berfungsi melindungi dan menjaga kelembutan kulit. Kulit berminyak bisa semakin parah akibat perubahan hormon, pertambahan usia, dan lain-lain. Pemilik kulit dengan tipe berminyak biasanya bermasalah dengan pori-pori besar, jerawat, dan komedo. Mengutip American Academy Dermatology Association, seseorang dengan kulit berminyak umumnya memiliki kulit yang tebal dan lebih sedikit keriput. Pemilik kulit berminyak disarankan untuk mencuci muka dua kali sehari, menghindari pembersih scrub, dan memilih produk perawatan kulit berlabel nonkomedogenik.",
    "Kering": "Kulit kering biasanya disebabkan oleh faktor luar seperti udara yang kering, kebiasaan mandi terlalu lama, dan paparan bahan kimia pada produk pembersih kulit. Jenis kulit ini juga bisa dimiliki oleh orang yang mengalami perubahan hormon atau mulai menua. Ciri khas dari kulit kering adalah pori-pori yang kecil, adanya bercak kemerahan, dan penampilan kulit yang cenderung kusam. Kulit yang sangat kering dapat menjadi kasar, pecah-pecah, serta mudah gatal dan teriritasi. Kulit kering yang tidak terawat dapat mengalami peradangan dan bahkan berkembang menjadi eksim.",
    "Normal": "Kulit normal tidak terlalu kering maupun berminyak. Tipe kulit ini memiliki kelembapan dan kekenyalan yang cukup karena sebum minyak alami kulit tersebar merata, tapi produksi minyak tidak berlebihan sehingga kulit tidak tampak mengilap. Kulit normal hanya memiliki sedikit masalah kulit atau kadang tidak ada sama sekali. Kulit tidak tampak kusam, memiliki persebaran warna yang merata, dan pori-pori yang tidak terlalu besar. Tipe kulit ini juga tidak mudah mengalami iritasi. Selain itu, tipe kulit wajah ini biasanya jarang mengalami masalah kulit, seperti komedo atau jerawat. Perawatan untuk kulit normal juga cenderung lebih mudah.",
    "Kombinasi": "Kulit kombinasi adalah perpaduan dari beberapa tipe kulit dan merupakan tipe kulit yang paling umum. Ciri khasnya adalah area tertentu pada kulit terasa berminyak, sedangkan area yang lain justru normal, kering, atau bahkan sensitif. Bagian kulit yang biasanya berminyak adalah T-Zone yang terdiri dari dahi, hidung, dan dagu. Sementara itu, area kulit yang kering adalah di sekitar mata dan mulut. Pipi bisa kering ataupun berminyak, tergantung seberapa banyak produksi sebum. Pemilik kulit kombinasi juga menghadapi masalah yang sama dengan kulit berminyak, yakni pori-pori wajah besar, komedo, dan kulit yang tampak mengilap."
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
        .type {
            font-size: 17px;
            color: #ffffff;
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
st.sidebar.title("Menu")
menu = st.sidebar.radio("", ["Home", "Info"])

if menu == "Home":
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

                    st.image(face_image, use_column_width=True)
                    st.markdown(f'**<p class="type">Jenis Kulit Wajah: {predicted_skin_type}</p>**', unsafe_allow_html=True)
                    st.markdown(f'<p class="result">{skin_type_descriptions[predicted_skin_type]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="care">{skin_type_care[predicted_skin_type]}</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="description">Wajah tidak terdeteksi. Silahkan ambil gambar lagi.</p>', unsafe_allow_html=True)

    take_photo()

elif menu == "Info":
    st.markdown('<h1 class="title">✨Tipe Kulit Wajah✨</h1>', unsafe_allow_html=True)
    selected_skin_type = st.selectbox("Pilih tipe kulit wajah dibawah untuk melihat informasi lebih lanjut:", skin_types)
    image_path = f'images/{selected_skin_type}.jpg'
    st.image(image_path,  width=300)
    st.markdown(f'### {selected_skin_type}')
    st.markdown(f'<div style="text-align: justify;">{skin_type_info[selected_skin_type]}</div>', unsafe_allow_html=True)
    st.markdown("---")
