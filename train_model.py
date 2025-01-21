import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.preprocessing import LabelEncoder

# Folder dataset yang sudah dikumpulkan
dataset_folder = 'dataset'

# Mengumpulkan data dan label
def create_dataset():
    data = []
    labels = []
    
    for letter in os.listdir(dataset_folder):  # Iterasi setiap folder huruf
        letter_folder = os.path.join(dataset_folder, letter)
        if os.path.isdir(letter_folder):
            for image_name in os.listdir(letter_folder):  # Iterasi setiap gambar
                image_path = os.path.join(letter_folder, image_name)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Konversi gambar ke grayscale
                image = cv2.resize(image, (64, 64))  # Ukuran standar
                data.append(image.flatten())  # Ubah gambar ke vektor 1D
                labels.append(letter)  # Tambahkan label
    
    #Konversi data dan label menjadi numpy array
    data = np.array(data)
    labels = np.array(labels)
    
    #Encode label huruf menjadi angka
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    return data, labels, label_encoder

# Melatih model KNN
def train_model():
    print("Mengumpulkan data untuk pelatihan...")
    data, labels, label_encoder = create_dataset()

    # Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Membuat dan melatih model KNN
    # KNN menggunakan 3 tetangga terdekat
    model = KNeighborsClassifier(n_neighbors = 3 )
    model.fit(X_train, y_train)

    # Simpan model dan label encoder
    with open('gesture_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"Model dilatih dan disimpan. Akurasi: {model.score(X_test, y_test)}")

if __name__ == "__main__":
    train_model()
    print("Model telah dilatih dan disimpan.")