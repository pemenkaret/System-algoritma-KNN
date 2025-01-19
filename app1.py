from flask import Flask, render_template, request, jsonify, url_for
import os
import pickle
import cv2

app = Flask(__name__)

# Folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = './uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Membuat folder uploads jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Memuat model dan label encoder
with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Fungsi untuk memproses gambar dan membuat prediksi
def predict_image(image_path):
    # Membaca dan memproses gambar
    image = cv2.imread(image_path) 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Konversi gambar ke grayscale
    resized_image = cv2.resize(gray_image, (64, 64)) # Ukuran standar
    features = resized_image.flatten().reshape(1, -1) # Ubah gambar ke vektor 1D
    
    # Prediksi menggunakan model
    prediction = model.predict(features)
    predicted_letter = label_encoder.inverse_transform(prediction)
    return predicted_letter[0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Memeriksa apakah ada file yang diunggah
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    
    file = request.files['file']
    if file:
        # Simpan gambar yang diunggah
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        
        # Prediksi huruf berdasarkan gambar
        predicted_letter = predict_image(image_path)
        
        # Return hasil prediksi
        return jsonify({
            'predicted_letter': predicted_letter,
            'image_url': url_for('static', filename=f'uploads/{file.filename}')
        })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
