from flask import Flask, render_template, request, jsonify
import os
import pickle
import cv2

app = Flask(__name__)

# Tentukan folder untuk menyimpan gambar yang di-upload
UPLOAD_FOLDER = './uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Membuat folder uploads jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Memuat model dan label encoder
with open('model/gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Fungsi untuk memproses gambar dan membuat prediksi
def predict_image(image_path):
    # Membaca gambar dan mengubahnya menjadi format yang sesuai untuk model
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (64, 64))
    features = resized_image.flatten().reshape(1, -1)
    
    # Prediksi menggunakan model
    prediction = model.predict(features)
    predicted_letter = label_encoder.inverse_transform(prediction)
    return predicted_letter[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        # Simpan gambar yang di-upload
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        
        # Prediksi huruf berdasarkan gambar
        predicted_letter = predict_image(image_path)
        
        # Mengembalikan hasil prediksi dan path gambar dalam format JSON
        return jsonify({'predicted_letter': predicted_letter, 'image_path': image_path})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
