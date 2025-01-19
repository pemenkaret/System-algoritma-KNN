# Gesture Recognition System

Sistem ini digunakan untuk mengenali gambar tangan yang menunjukkan huruf tertentu dan memprediksi huruf yang sesuai menggunakan model **K-Nearest Neighbors (KNN)**.
## Deskripsi
Program ini adalah aplikasi pengenalan gestur tangan yang merepresentasikan huruf tertentu menggunakan algoritma **K-Nearest Neighbors (KNN)**. Dataset gambar tangan diklasifikasikan ke dalam huruf alfabet berdasarkan fitur citra.

## Cara Kerja Program
1. **Pengumpulan Dataset**:
   - Dataset berupa gambar tangan yang dikelompokkan ke dalam folder sesuai label huruf (contoh: `A`, `B`, `C`).
   - Setiap gambar diubah menjadi **grayscale**, diubah ukurannya menjadi `(64x64)` piksel, lalu diratakan menjadi vektor 1D.

2. **Pelatihan Model**:
   - Dataset dibagi menjadi **training data (80%)** dan **testing data (20%)**.
   - Model KNN dilatih menggunakan parameter:
     - `n_neighbors=3`: Model menggunakan 3 tetangga terdekat untuk prediksi.
   - Model dan encoder label disimpan dalam file `gesture_model.pkl` dan `label_encoder.pkl`.

3. **Prediksi Gambar**:
   - Gambar baru diunggah melalui antarmuka API atau UI.
   - Gambar diproses (grayscale, resize, flatten) lalu diprediksi menggunakan model KNN.
   - Hasil prediksi berupa huruf yang sesuai dengan gestur tangan.

## Algoritma K-Nearest Neighbors (KNN)

### Cara Kerja KNN
**KNN** adalah algoritma pembelajaran mesin berbasis **instance-based learning** yang bekerja sebagai berikut:
1. Saat pelatihan:
   - Data dilatih dengan menyimpan fitur gambar dan label ke dalam training set.
   - Tidak ada generalisasi model dilakukan (lazy learning).
2. Saat prediksi:
   - Untuk setiap gambar baru, algoritma menghitung **jarak Euclidean** antara gambar baru dan seluruh gambar dalam training set.
   - Gambar diklasifikasikan ke kelas berdasarkan **mayoritas label dari k tetangga terdekat**.

### Parameter Penting
- `n_neighbors=3`: Jumlah tetangga terdekat yang dipertimbangkan.
- **Jarak Euclidean**: Digunakan untuk mengukur kedekatan antara dua gambar.

### Kelebihan KNN
- **Sederhana dan Efektif**: Tidak memerlukan asumsi distribusi data.
- **Fleksibel**: Dapat digunakan untuk berbagai jenis data, termasuk gambar, teks, atau numerik.
- **Interpretasi Mudah**: Prediksi didasarkan pada data yang ada tanpa transformasi kompleks.

### Kekurangan KNN
- **Efisiensi**: Membutuhkan lebih banyak waktu untuk prediksi karena harus menghitung jarak terhadap seluruh training set.
- **Sensitif terhadap Skala Fitur**: Normalisasi data penting untuk menghindari fitur tertentu mendominasi.
- **Overfitting**: Jika jumlah tetangga terlalu kecil, model bisa menjadi terlalu sensitif terhadap noise.

## Alur Program
1. **Training**:
   - Dataset dikumpulkan dan diolah menjadi fitur numerik.
   - Model KNN dilatih dan disimpan sebagai file pickle.
2. **Prediction**:
   - Gambar baru diunggah melalui API.
   - Gambar diproses dan diprediksi menggunakan model KNN.
   - Prediksi dikembalikan dalam bentuk huruf.

## Cara Menjalankan Program
1. **Persiapkan Dataset**:
   - Letakkan dataset gambar dalam folder `dataset`, dengan subfolder berdasarkan label huruf.
2. **Latih Model**:
   - Jalankan script `train_model.py` untuk melatih model KNN.
3. **Jalankan Server**:
   - Jalankan `app.py` untuk menjalankan server Flask.
4. **Unggah Gambar**:
   - Gunakan endpoint `/predict` untuk mengunggah gambar dan mendapatkan hasil prediksi.

## File dan Folder
- `dataset/`: Folder berisi dataset gambar.
- `train_model.py`: Script untuk melatih model.
- `gesture_model.pkl`: File pickle model KNN.
- `label_encoder.pkl`: File encoder label.
- `app.py`: Script utama untuk server Flask.

## Teknologi yang Digunakan
- Python
- OpenCV (Pengolahan gambar)
- Flask (Server API)
- Scikit-learn (KNN)

-----

# Proses Prediksi
### *Input:*
- Gambar tangan (contoh: huruf "A") diunggah melalui antarmuka sistem.

### **Proses:**
1. **Konversi ke Grayscale:**
   - Gambar diubah menjadi format grayscale untuk menghilangkan informasi warna yang tidak relevan.
2. **Resize:**
   - Gambar diubah ukurannya menjadi 64x64 piksel untuk standarisasi.
3. **Flatten:**
   - Gambar diratakan menjadi vektor 1 dimensi sebagai input ke model.
4. **Prediksi:**
   - Model **KNN** memproses vektor gambar dan memprediksi huruf berdasarkan fitur gambar.

### *Output:*
- Huruf yang diprediksi: **"A"**.

### **Response JSON:**
Jika prediksi berhasil, sistem akan mengembalikan response dalam format JSON seperti berikut:

```json
{
  "predicted_letter": "A",
  "image_url": "/static/uploads/input_image.jpg"
}
