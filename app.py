from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from email.message import EmailMessage
from PIL import Image, ImageOps
from imutils.video import FPS
from datetime import datetime
from st_aggrid import AgGrid
import tensorflow.keras
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import sqlite3
import imutils
import smtplib
import imghdr
import base64
import time
import cv2
import io 
import os
import os.path
import pathlib








st.set_page_config( 
layout="wide",  
initial_sidebar_state="auto",
page_title= "Absen CNN",
)


# db management
import sqlite3
conn = sqlite3.connect("data.db")
c = conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS usertable(username TEXT, password TEXT)')

def add_userdata(username, password):
    c.execute('INSERT INTO usertable(username, password) VALUES (?,?)', (username, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM usertable WHERE username =? AND password =?', (username, password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM usertable')
    data = c.fetchall()
    return data

def main():
    menu = ("Masuk", "Daftar")
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Masuk":
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type= "password")
        if st.sidebar.checkbox("Login"):
            create_usertable()
            result = login_user(username, password)
            if result:
                st.success("Masuk sebagai {}".format(username))
                choose = option_menu("Menu", ["Home", "Train", "Absen Wajah", "Data Absen", "Tentang"],
                        icons=['house', 'camera fill', 'eye','book','person-circle'],
                        menu_icon="app-indicator", 
                        default_index=0, 
                        orientation="horizontal",
                        styles={
                            "container": {"padding": "5!important", "background-color": "#1C1C1C"},
                            "icon": {"color": "white", "font-size": "20px"}, 
                            "nav-link": {"color": "white","font-size": "14px", "text-align": "center", "margin":"0px", "--hover-color": "#1F7AFF"},
                            "nav-link-selected": {"background-color": "#19286D"},
                        }
                )
                if choose == "Home":
                    profile = Image.open("./Images/logo.png")
                    col1, col2 = st.columns( [0.8, 0.2])
                    with col1:
                        st.write("")
                        st.header("Sistem Deteksi Wajah Untuk Pencatatan Kehadiran Mahasiswa Di Kelas Menggunakan Metode Convolutional Neural Network")
                    with col2: 
                        st.write("")
                        st.image(profile, width=150 )
                        
                # Menu train
                elif choose == "Train":
                    st.title("Proses Training") 
                    st.header("Tahapan pada proses training untuk membuat dataset menjadi model untuk deteksi adalah sebagai berikut")
                    
                    
                    #Tahap persiapan
                    st.subheader("Tahap persiapan")
                    st.write("Proses training dilakukan manual dan offline menggunakan python dikarenakan komputasi yang berat pada proses training")
                    st.write("Silahkan terlebih dahulu download template folder dan file yang diperlukan pada link [Template Folder](https://www.python.org/downloads/)")
                    st.write("Download [Anaconda](https://repo.anaconda.com/archive/) direkomendasikan Anaconda3-2019.07")                    
                    st.write("Install semua Library Python yang ada dibawah")
                    df_lib = pd.read_csv("lib.csv")
                    st.write(df_lib)
                    def convert_df_lib(df_lib):
                        return df_lib.to_csv().encode('utf-8') 
                    csv = convert_df_lib(df_lib)
                    st.write("Install Library Python dengan cara ketik command dibawah ini di terminal/cmd")
                    st.code("pip install nama_library==versi", language='python')
                    st.markdown("sebagai contoh kita akan meng-install library **Tensorflow** dengan versi **2.9.1** maka command yang kita ketik di terminal/cmd adalah :")
                    st.code("pip install tensorflow==2.9.1", language='python')
                    st.write("")


                    #Pengumpulan dataset wajah
                    st.subheader("Pengumpulan dataset wajah")
                    st.write("Citra wajah bisa diambil dengan 2 (dua) cara yaitu :")
                    st.markdown("**1. Secara manual menggunakan kamera handphone atau DSLR**")
                    st.markdown("Cara ini bisa dilakukan dengan menggunakan kamera handphone atau DSLR kemudian gambar mahasiswa yang diambil akan dibuat folder secara manual dengan ketentuan nama folder **NAMA_NIM** pada path folder **Data>No_Cropped** yang sudah disediakan di template folder seperti pada GIF dibawah ini")
                    file_manual = open("./Images/dataset_folder_manual.gif", "rb")
                    contents = file_manual.read()
                    data_url = base64.b64encode(contents).decode("utf-8")
                    file_manual.close()
                    st.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" alt="dataset_folder_manual">',
                        unsafe_allow_html=True,
                    )
                    st.caption("Dimana folder NAMA_NIM berisikan kumpulan citra wajah mahasiswa")            
                    image = Image.open('Images/nama_nim_folder_manual.png')
                    st.image(image, caption='contoh isi folder NAMA_NIM')
                    st.markdown("citra wajah yang diambil dari handphone/kamera diletakkan pada folder **No_Cropped** dikarenakan dataset akan di crop pada bagian wajah pada keseluruhan folder menggunakan **Auto_Crop.ipynb** ayng ada pada **Template Folder**. Crop pada gambar bagian wajah diperlukan untuk meningkatkan akurasi training nantinya")
                    st.markdown("setelah data citra wajah di crop otomatis pada bagian muka menggunakan **Auto_Crop.ipynb** maka akan ter-generate folder baru dengan nama yang sama pada folder **Cropped** seperti GIF dibawah ini")
                    auto_crop = open("./Images/auto_crop.gif", "rb")
                    contents = auto_crop.read()
                    data_url = base64.b64encode(contents).decode("utf-8")
                    auto_crop.close()
                    st.markdown(
                        f'<img src="data:image/gif;base64,{data_url}" alt="Folder tergenerate secara otomatis">',
                        unsafe_allow_html=True,
                    )                    
                    st.caption("Folder tergenerate secara otomatis sesuai dengan nama folder yang ada di folder No_Cropped") 
                    st.write("")
                    st.markdown("**2. Menggunakan Webcam**")
                    st.markdown("Pada Template Folder terdapat file python dengan nama **get_data.py**, file python ini fungsinya untuk mengambil gambar mahasiswa yang berada didepan webcam kemudian akan tersimpan secara otomatis pada folder **Data>Cropped>NAMA_NIM** dikarenakan pada file python secara otomatis setelah mengambil wajah mahasiswa akan meng-crop pada bagian wajah saja.")
                    st.markdown("untuk bisa menjalankan **get_data.py** cukup ketik perintah dibawah ini pada terminal/cmd pada folder **Template Folder**")
                    st.code("python get_data.py", language='python')
                    st.write("")
                    
                    
                    #Tahap training
                    st.subheader("Tahap Training")
                    st.write("Pada tahap ini dataset yang telah disediakan akan diproses melalui program training yang telah dibuat.")
                    st.markdown("**Import Library**")
                    lib_import_code = '''from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import os'''
                    st.code(lib_import_code, language='python')
                    
                    st.markdown("**Menentukan nilai learning rate, epoch dan batch size**")
                    lrebs_code = '''INIT_LR = 1e-4
EPOCHS = 1000
BS = 32'''
                    st.code(lrebs_code, language='python')
                    
                    st.markdown("**Mengambil dataset dari folder**")
                    getdataset_code = '''imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []'''
                    st.code(getdataset_code, language='python')
                    
                    st.markdown("**Mengekstrak label class dari nama nama per-folder, preprocessing gambar menjadi ukuran 224x224px dan mengubahnya menjadi array, mengupdate data dan label**")
                    extract_code = '''for imagePath in imagePaths:
    # ekstrak label class dari filename
    label = imagePath.split(os.path.sep)[-2]

    # memuat input gambar (224x224) dan melakukan preprocessing
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # update data dan label list
    data.append(image)
    labels.append(label)'''
                    st.code(extract_code, language='python')
                    
                    st.markdown("**Konversi data dan label menjadi numpy array, encoding**")
                    convert_code = '''# konversi data dan label ke numpy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)'''
                    st.code(convert_code, language='python')
                    
                    st.markdown("**Partisi dataset ke training dan testing.**")
                    split_code = '''# parsisi data ke training dan testing, 75% training  dan 25% testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)'''
                    st.code(split_code, language='python')
                    
                    st.markdown("**Augmentasi data citra agar menjadi lebih variatif, yang bertujuan untuk meningkatkan akurasi.**")
                    augment_code = '''aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")'''
                    st.code(augment_code, language='python')
                    
                    st.markdown("**Proses training menggunakan Algoritma CNN dan arsitektur MobileNetV2**")
                    algorithm_code = '''headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(10, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False'''
                    st.code(algorithm_code, language='python')
                    
                    st.markdown("**Proses compile model, dan menampilkan info train head dari network.**")
                    compile_code = '''# compile model
print("[INFO] meng-compile model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train head dari network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)'''
                    st.code(compile_code, language='python')
                    
                    st.markdown("**Membuat predikisi untuk testing set dan menampilkan laporan klasifikasi.**")
                    predict_code = '''# buat prediksi untuk testing set
print("[INFO] mengevaluasi network...")
predIdxs = model.predict(testX, batch_size=BS)

# untuk setiap gambar di testing set harus ditemukan index
# dari label telebih dahulu dengan hasil prediksi paling besar
predIdxs = np.argmax(predIdxs, axis=1)

# menampilkan laporan klasifikasi
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))'''
                    st.code(predict_code, language='python')
                    
                    st.markdown("**Menyimpan model ke disk dengan format .h5**")
                    save_code = '''print("[INFO] menyimpan model deteksi masker...")
model.save(args["model"], save_format="h5")'''
                    st.code(save_code, language='python')
                    
                    st.markdown("**Menampilkan plot hasil training.**")
                    plot_code = '''# plot untuk training loss dan accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss dan Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])'''
                    st.code(plot_code, language='python')
                    st.write("")
                    
                    image = Image.open('./Images/contoh_plot.png')
                    st.image(image, caption='contoh plot yang akan dihasilkan')
                    
                    
                    
                # Menu absen
                elif choose == "Absen Wajah":
                    st.spinner('Memuat model deteksi wajah...')
                    st.title("Absen Wajah")
                    
                    def absen_jika_hadir(name):
                        with open('absensi.csv', 'r+') as f:
                            myDataList = f.readlines()
                            nameList = []
                            for line in myDataList:
                                entry = line.split(',')
                                nameList.append(entry[0])
                            if name not in nameList:
                                now = datetime.datetime.now()
                                dtString = now.strftime('%H:%M:%S')
                                tanggal = now.strftime('%x')
                                ket = ('Hadir')
                                f.writelines(f'\n{name},{tanggal},{dtString},{ket}')

                    def detect_and_predict_mask(frame, faceNet, maskNet):
                        # ambil dimensi dari frame dan construct blob
                        (h, w) = frame.shape[:2]
                        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

                        # umpan blob dan ambil deteksi wajah
                        faceNet.setInput(blob)
                        detections = faceNet.forward()

                        # inisialiasi daftar wajah, lokasi dan daftar prediksi dari jaringan deteksi masker
                        faces = []
                        locs = []
                        preds = []

                        # loop deteksi
                        for i in range(0, detections.shape[2]):
                            # ekstrak nilai confidence (perkiraan/peluang) yang berhubungan dengan deteksi
                            confidence = detections[0, 0, i, 2]

                            # melakukan filter untuk deteksi yang dinilai lemah dan memastikan nilai confidence lebih besar dari nilai minimum
                            if confidence > 0.70:
                                # menghitung koordinat (x,y) dari bounding box untuk objek yang dideteksi
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")

                                # memastikan bounding box tepat berada pada dimensi frame
                                (startX, startY) = (max(0, startX), max(0, startY))
                                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                                #  ekstrak ROI wajah, konversi dari BRG ke channel RGB, resize ke 244x244 px dan lakukan prepocess
                                face = frame[startY:endY, startX:endX]
                                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                                face = cv2.resize(face, (224, 224))
                                face = img_to_array(face)
                                face = preprocess_input(face)

                                # menambahkan wajah dan bounding box ke respective list
                                faces.append(face)
                                locs.append((startX, startY, endX, endY))

                        # hanya membuat prdiksi jika minimal ada 1 wajah
                        if len(faces) == 1:
                            # deteksi wajah lebih dari 1 pada saat bersamaan
                            faces = np.array(faces, dtype="float32")
                            preds = maskNet.predict(faces, batch_size=32)

                        # return a 2-tuple of the face locations and their corresponding
                        # locations
                        return (locs, preds)


                    # memuat model deteksi wajah pada disk (ssd)                    
                    prototxtPath = ('Models/face_detector/deploy.prototxt')
                    weightsPath = ('Models/face_detector/res10_300x300_ssd_iter_140000.caffemodel')
                    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
                    maskNet = load_model('Models/models.model')


                    # inisialiasasi video stream dari kamera                    
                    
                    vs = cv2.VideoCapture(0)
                    run = st.checkbox('Buka Kamera')
                    time.sleep(2.0)
                    fps = FPS().start()
                    FRAME_WINDOW = st.image([])


                    # loop frame dari video stream
                    while run:
                        _, frame = vs.read()#
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#

                        # deteksi wajah pada frame menggunakan masker atau tidak
                        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                        # looping pada wajah lokasi wajah yang terdeteksi
                        for (box, pred) in zip(locs, preds):
                            # unpack bounding box dan prediksi
                            (startX, startY, endX, endY) = box
                            (DEVINA_HUMAIRA_PUTRI_1857301065, IKA_WULANDARI_1857301009, KHAIRUL_AZMAN_1857301038, MUHAMMAD_REZEKI_ANANDA_1857301017,
                            MUHAMMAD_RIZAL_1857301042, MUZAMMIL_1857301068, NURUL_FATANI_1857301060,
                            RAUDHY_1857301067, SALSABILA_AKMAL_1857301014, TAJUN_NUR_1857301053, SYIFA_ZAHRAH_1857301057) = pred

                            # label kelas yang akan ditampilkan dan warna dari bounding box yang akan digunakan
                            if DEVINA_HUMAIRA_PUTRI_1857301065>=0.70:
                                label = "Devina"
                                color = (0, 255, 0)
                                absen_jika_hadir('DEVINA_HUMAIRA_PUTRI_1857301065')

                            elif IKA_WULANDARI_1857301009>=0.70:
                                label = "Ika"
                                color = (0, 255, 0)
                                absen_jika_hadir('IKA_WULANDARI_1857301009')

                            elif  KHAIRUL_AZMAN_1857301038>=0.70:
                                label = "khairul"
                                color = (0, 255, 0)
                                absen_jika_hadir('KHAIRUL_AZMAN_1857301038')
                            
                            elif MUHAMMAD_RIZAL_1857301042>=0.70:
                                label = "rizal"
                                color = (0, 255, 0)
                                absen_jika_hadir('MUHAMMAD_RIZAL_1857301042')
                            
                            elif MUZAMMIL_1857301068>=0.70:
                                label = "muzammil"
                                color = (0, 255, 0)
                                absen_jika_hadir('MUZAMMIL_1857301068')
                            
                            elif NURUL_FATANI_1857301060>=0.70:
                                label = "fatani"
                                color = (0, 255, 0)
                                absen_jika_hadir('NURUL_FATANI_1857301060')
                            
                            elif RAUDHY_1857301067>=0.70:
                                label = "raudhy"
                                color = (0, 255, 0)
                                absen_jika_hadir('RAUDHY_1857301067')
                            
                            elif  SALSABILA_AKMAL_1857301014>=0.70:
                                label = "salsa"
                                color = (0, 255, 0)
                                absen_jika_hadir('SALSABILA_AKMAL_1857301014')
                            
                            elif  TAJUN_NUR_1857301053>=0.70:
                                label = "nur"
                                color = (0, 255, 0)
                                absen_jika_hadir('TAJUN_NUR_1857301053')
                                
                            elif  MUHAMMAD_REZEKI_ANANDA_1857301017>=0.70:
                                label = "Ananda"
                                color = (0, 255, 0)
                                absen_jika_hadir('MUHAMMAD_REZEKI_ANANDA_1857301017')
                            
                            elif  SYIFA_ZAHRAH_1857301057>=0.70:
                                label = "Syifa"
                                color = (0, 255, 0)
                                absen_jika_hadir('SYIFA_ZAHRAH_1857301057')
                            # if max(pred)==DEVINA_HUMAIRA_PUTRI_1857301065:
                            #     label = ("{}:".format (max(pred)))
                            #     color = (0, 255, 0)
                            #     absen_jika_hadir('DEVINA_HUMAIRA_PUTRI_1857301065')


                            # elif  max(pred)==IKA_WULANDARI_1857301009:
                            #     label = ("{}:".format (max(pred)))
                            #     color = (0, 0, 255)
                            #     absen_jika_hadir('IKA_WULANDARI_1857301009')

                            # elif  max(pred)==KHAIRUL_AZMAN_1857301038:
                            #     label = "khairul"
                            #     color = (0, 0, 255)
                            #     absen_jika_hadir('KHAIRUL_AZMAN_1857301038')
                        
                            
                            # elif  max(pred)==MUHAMMAD_RIZAL_1857301042:
                            #     label = "rizal"
                            #     color = (0, 0, 255)
                            #     absen_jika_hadir('MUHAMMAD_RIZAL_1857301042')
                            
                            # elif  max(pred)==MUZAMMIL_1857301068:
                            #     label = "muzammil"
                            #     color = (0, 0, 255)
                            #     absen_jika_hadir('MUZAMMIL_1857301068')
                            
                            # elif  max(pred)==NURUL_FATANI_1857301060:
                            #     label = "fatani"
                            #     color = (0, 0, 255)
                            #     absen_jika_hadir('NURUL_FATANI_1857301060')
                            
                            # elif  max(pred)==RAUDHY_1857301067:
                            #     label = "raudhy"
                            #     color = (0, 0, 255)
                            #     absen_jika_hadir('RAUDHY_1857301067')
                            
                            # elif  max(pred)==SALSABILA_AKMAL_1857301014:
                            #     label = "salsa"
                            #     color = (0, 0, 255)
                            #     absen_jika_hadir('SALSABILA_AKMAL_1857301014')
                            
                            # elif  max(pred)==TAJUN_NUR_1857301053:
                            #     label = "nur"
                            #     color = (0, 0, 255)
                            #     absen_jika_hadir('TAJUN_NUR_1857301053')
                                
                            # elif  max(pred)==MUHAMMAD_REZEKI_ANANDA_1857301017:
                            #     label = "Ananda"
                            #     color = (0, 0, 255)
                            #     absen_jika_hadir('MUHAMMAD_REZEKI_ANANDA_1857301017')
                            
                            else:
                                label = "Tidak dikenali"
                                color = (255, 0, 0)
                                
                            
                                
                            cv2.putText(frame, label, (startX-50, startY - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        FRAME_WINDOW.image(frame)
                    
                # Menu upload
                elif choose == "Data Absen":
                    st.title("Data Absen")
                    df = pd.read_csv("absensi.csv")
                    AgGrid(df)
                    def convert_df(df):
                        return df.to_csv().encode('utf-8') 
                    csv = convert_df(df)
                    col1, col2 = st.columns( [0.5, 0.5])
                    with col1:
                        st.download_button(
                            "Download Data Absen",
                            csv,
                            "absensi_download.csv",
                            "text/csv",
                            key='download-csv'
                            )
                    
                    with col2:
                        uploaded_file = st.file_uploader("Pilih file absen")
                        if uploaded_file is not None:
                            bytes_data = uploaded_file.getvalue()
                            data = uploaded_file.getvalue().decode('utf-8').splitlines()         
                        option1 = st.selectbox(
                                    'Pilih Kelas:',
                                    ('TI-1A', 'TI-1B', 'TI-1C', 'TI-1D', 'TI-2A','TI-2B', 'TI-2C', 'TI-3A', 'TI-3B', 'TI-3C', 'TI-4A', 'TI-4B', 'TI-4C'))
                        
                        option2 = st.selectbox(
                                    'Pilih MK:',
                                    ('KONSEP PEMROGRAMAN', 'P-KONSEP PEMROGRAMAN', 'KONSEP TEKNOLOGI INFORMASI', 'KONSEP BASIS DATA', 'P-KONSEP BASIS DATA', 'LOGIKA DAN ALGORITMA', 'MATEMATIKA TEKNIK', 'P-KETERAMPILAN KOMPUTER', 'ENGLISH FOR ACADEMIC LISTENING', 'METODE NUMERIK', 'P-METODE NUMERIK', 'PEMROGRAMAN BERORIENTASI OBJEK', 'P-PEMROGRAMAN BERORIENTASI OBJEK', 'KONSEP J-KOMPUTER', 'P-KONSEP J KOMPUTER', 'REKAYASA PERANGKAT LUNAK', 'P-PEMROGRAMAN WEB LANJUT','PEMROGRAMAN WEB LANJUT', 'ENGLISH FOR ACADEMIC READING', 'PEMROGRAMAN MOBILE', 'P-PEMROGRAMAN MOBILE', 'KEAMANAN JARINGAN KOMPUTER', 'P-KEAMANAN JARINGAN KOMPUTER', 'SISTEM INFORMASI MANAJEMEN DAN SPK', 'STATISTIK DAN PROBABILITAS', 'PENGOLAHAN CITRA DIGITAL', 'P-PENGOLAHAN CITRA DIGITAL', 'RANCANGAN ANALISA ALGORITMA', 'WORKSHOP PENGEMBANGAN PERANGKAT LUNAK', 'BHS INDONESIA', 'PRAKTIKUM KUALITAS PL', 'PENGUJIAN KUALITAS PL', 'DATA MINING & WAREHOUSE', 'INTERAKSI MANUSIA &KOMPUTER', 'GEOGRAFIS INFORMATION SYSTEM', 'P-GEOGRAFIS INFORMATION SYSTEM', 'P-BIG DATA', 'BIG DATA', 'P-KOMPUTASI CLOUD', 'KOMPUTASI CLOUD'))
                        
                        def upload():
                            if uploaded_file is None:
                                st.session_state["upload_state"] = "Upload file terlebih dahulu!"
                            else:
                                data = uploaded_file.getvalue().decode('utf-8')
                                parent_path = pathlib.Path(__file__).parent.parent.resolve()          
                                save_path = os.path.join(parent_path, "absensi", option1, option2)
                                complete_name = os.path.join(save_path, uploaded_file.name)
                                destination_file = open(complete_name, "w")
                                destination_file.write(data)
                                destination_file.close()
                        st.button("Upload", on_click=upload)
                        
                    
                # Menu tentang
                elif choose == "Tentang":
                    col1, col2 = st.columns( [0.8, 0.2])
                    with col1:
                        st.title("Biodata") 
                    with col2: 
                        st.image("./Images/foto_about.jpeg", width=200 )


                    
    elif choice == "Daftar":
        st.write("")
        st.write("")
        st.subheader("👈 Silahkan daftar untuk mengakses aplikasi")
        profile = Image.open("./Images/logo.png")
        col1, col2 = st.columns( [0.8, 0.2])
        with col1:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.header("Sistem Deteksi Wajah Untuk Pencatatan Kehadiran Mahasiswa Di Kelas Menggunakan Metode Convolutional Neural Network")   
            st.write("")
            
        with col2: 
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.image(profile, width=150 )
        new_username = st.sidebar.text_input("User Name")
        new_password = st.sidebar.text_input("Password", type= "password")

        if st.sidebar.button("Daftar"):
            create_usertable()
            add_userdata(new_username, new_password)
            st.balloons()
            st.success("Anda telah berhasil membuat akun")
            st.info("Buka Menu Masuk untuk Masuk")


if __name__ == "__main__":
    main()