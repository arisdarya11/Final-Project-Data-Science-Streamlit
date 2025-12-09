# Predicting Employee Analytics: Turnover with Machine Learning

Proyek ini bertujuan untuk memprediksi kemungkinan karyawan melakukan resign (turnover) berdasarkan data historis SDM menggunakan pendekatan data science dan machine learning.

---

## Business Understanding

### Latar Belakang

Turnover karyawan merupakan salah satu tantangan terbesar dalam pengelolaan sumber daya manusia di berbagai organisasi. Tingginya tingkat pengunduran diri tidak hanya meningkatkan biaya rekrutmen dan pelatihan, tetapi juga berdampak pada produktivitas tim, kontinuitas operasional, serta kualitas layanan perusahaan.

Banyak perusahaan mengalami kesulitan dalam mengidentifikasi karyawan yang berpotensi meninggalkan organisasi karena keputusan resign dipengaruhi oleh berbagai faktor yang saling terkait, seperti tingkat kepuasan kerja, beban kerja, masa kerja, lingkungan kerja, dan peluang pengembangan karier.

Oleh karena itu, diperlukan pendekatan berbasis data (data-driven) untuk membantu perusahaan memprediksi risiko turnover secara lebih akurat. Proyek ini memanfaatkan teknik data science dan machine learning untuk menganalisis pola perilaku karyawan dan membangun model prediksi yang dapat membantu manajemen dalam mengambil keputusan strategis secara preventif dan berkelanjutan.

### Rumusan Masalah

Saat ini perusahaan belum memiliki sistem yang mampu mengidentifikasi secara dini karyawan yang berpotensi melakukan resign, sehingga menyulitkan manajemen dalam mengambil tindakan preventif yang tepat.

### Tujuan Proyek

Proyek ini bertujuan untuk:
- Mengidentifikasi faktor-faktor utama yang berpengaruh terhadap employee turnover.
- Membangun model prediksi untuk mengklasifikasikan karyawan yang berpotensi resign.
- Memberikan rekomendasi strategi dan kebijakan HR berbasis data untuk mengurangi tingkat turnover.

---

## Dataset

- Dataset HR_comma_sep.csv
https://www.kaggle.com/code/marinp/hr-comma-sep
- Data HR historis (numerik & kategorikal).
- Target variable: `left` (1 = resign, 0 = stay).

---

## Metodologi

Tahapan yang dilakukan dalam proyek ini meliputi:
- Data Cleaning dan Preprocessing.
- Exploratory Data Analysis (EDA).
- Feature Engineering.
- Training dan evaluasi model machine learning.
- Hyperparameter tuning.
- Model interpretability (feature importance).

---

## Model Machine Learning

Algoritma yang digunakan dalam proyek ini:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

## Evaluasi Model

Model dievaluasi menggunakan metrik:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## Hasil dan Insight Utama

### 1. Exploratory Data Analysis (EDA)

Ditemukan bahwa fitur `number_project`, `average_monthly_hours`, `time_spend_company`, dan `satisfaction_level` memiliki hubungan yang kuat dengan risiko turnover.

Karyawan dengan tingkat kepuasan rendah, beban kerja tinggi, dan masa kerja yang singkat cenderung memiliki potensi resign lebih tinggi.

### 2. Modeling

Beberapa algoritma diuji, seperti Logistic Regression, Decision Tree, Random Forest, dan XGBoost.

Model berbasis ensemble (Random Forest dan XGBoost) memiliki performa lebih baik dibandingkan model linear.

### 3. Hyperparameter Tuning

Hyperparameter tuning berhasil meningkatkan stabilitas dan performa model, terutama pada Random Forest dan XGBoost.

Risiko overfitting dapat dikurangi, meskipun masih terdapat perbedaan kecil antara performa data training dan testing.

### 4. Pemilihan Model Terbaik

Model XGBoost (tuned) dipilih sebagai model terbaik karena memiliki kombinasi metrik accuracy, recall, F1-score, dan ROC-AUC tertinggi pada data uji.

Turnover dipengaruhi oleh kombinasi faktor `number_project`, `average_monthly_hours`, dan `time_spend_company` yang juga berdampak pada tingkat kepuasan kerja.

Model ini menunjukkan kemampuan generalisasi yang baik dan layak digunakan untuk skenario dunia nyata.

---

## Rekomendasi Strategi HR

### 1. Pengaturan Beban Kerja (number_project)
- Hindari pemberian proyek yang terlalu banyak atau terlalu sedikit.
- Gunakan dashboard untuk memantau distribusi beban kerja.

### 2. Retensi Berdasarkan Masa Kerja (time_spend_company)
- Fokus pada karyawan dengan masa kerja 3–5 tahun.
- Sediakan career path yang jelas.
- Lakukan diskusi karier secara rutin.

### 3. Peningkatan Kepuasan Kerja (satisfaction_level)
- Lakukan survei kepuasan karyawan setiap kuartal.
- Prioritaskan perbaikan pada aspek yang paling banyak dikeluhkan.
- Terapkan program employee recognition.

### 4. Transparansi Evaluasi Kinerja (last_evaluation)
- Terapkan sistem feedback dua arah.
- Gunakan penilaian berbasis hasil dan proses kerja.
- Hindari kondisi high performer yang mengalami overwork.

### 5. Manajemen Jam Kerja (average_monthly_hours)
- Terapkan kebijakan lembur dengan persetujuan terlebih dahulu.
- Sediakan fleksibilitas kerja (WFH dan jam kerja fleksibel).
- Pantau jam kerja melalui sistem monitoring digital.

### 6. Intervensi Preventif untuk Karyawan Berisiko Tinggi
- Gunakan model prediksi untuk mendeteksi risiko resign.
- Lakukan stay interview secara berkala.
- Siapkan intervensi seperti pengurangan workload, mentoring, dan diskusi karier.

### 7. Review Kebijakan Promosi (promotion_last_5years)
- Tingkatkan transparansi kriteria promosi.
- Pastikan proses promosi adil dan berbasis kompetensi.

### 8. Strategi Kompensasi (salary)
- Lakukan benchmarking gaji secara berkala.
- Berikan insentif berbasis kinerja dan tunjangan yang kompetitif.

---

## Demo Aplikasi

Modern Machine Learning App for Employee Turnover Analysis powered by XGBoost:

https://turnover-prediction-app.streamlit.app

---


