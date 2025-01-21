import cv2
import dlib
import numpy as np
import imutils

# Load model deteksi wajah dan landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def calculate_symmetry(landmarks):
    """Menghitung tingkat simetri berdasarkan landmark wajah."""
    #dlib menyediakan model dengan 68 titik landmark pada wajah yang sudah dilabeli dan terstruktur dengan sangat spesifik. 
    #dengan menggunakan model ini, kita dapat memperoleh informasi tentang posisi dan orientasi wajah kita dengan lebih akurat.
    
    left_points = [landmarks.part(i) for i in range(17, 27)]  # Landmark sisi kiri indeks 17-26
    right_points = [landmarks.part(i) for i in range(0, 10)]  # Landmark sisi kanan indeks 0-9 (dibalik)

    # Mengonversi titik landmark menjadi array NumPy untuk perhitungan
    left_points = np.array([(p.x, p.y) for p in left_points])  #mengonversi menjadi array
    right_points = np.array([(p.x, p.y) for p in right_points[::-1]])  #urutan array Dibalik untuk kecocokan

    # Selisih jarak antara titik landmark sisi kiri dan kanan
    diff = np.abs(left_points - right_points)
    symmetry_score = np.mean(diff) #Hitung rata-rata selisih

    return symmetry_score

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() #membaca frame
    if not ret:
        break

    frame = imutils.resize(frame, width=600) # Atur lebar gambar
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Konversi gambar ke grayscale
    faces = detector(gray) # Deteksi wajah dari hasil konversi grayscale

    for face in faces:
        landmarks = predictor(gray, face)

        # Hitung skor simetri
        symmetry_score = calculate_symmetry(landmarks)

        # Tampilkan skor
        #jika score simstry semakin tinggi maka muka tidak simetrys
        cv2.putText(frame, f"Symmetry Score: {symmetry_score:.2f}", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
        #jika score simstry semakin rendah maka muka simetry
        if symmetry_score > 70:
            cv2.putText(frame, "Not Symmetrical", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    

        # Gambar landmark
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

# Tampilkan frame
    cv2.imshow("Face Symmetry Detection", frame)

# Tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()