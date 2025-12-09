---------Predicting Employee Analytics : Turnover with Machine Learning------------

This Project to predict the possibility of employees turning over (resigning) based on historical HR data.

BUSSINESS UNDERSTANDING :

LATAR BELAKANG 
Turnover karyawan merupakan salah satu tantangan terbesar dalam pengelolaan sumber daya manusia di berbagai organisasi. Tingginya tingkat pengunduran diri tidak hanya meningkatkan biaya rekrutmen dan pelatihan, tetapi juga berdampak pada produktivitas tim, kontinuitas operasional, serta kualitas layanan perusahaan.

Banyak perusahaan mengalami kesulitan dalam mengidentifikasi karyawan yang berpotensi meninggalkan organisasi, karena keputusan resign umumnya dipengaruhi oleh berbagai faktor yang saling terkait, seperti tingkat kepuasan kerja, beban kerja, masa kerja, lingkungan kerja, serta peluang pengembangan karier.

Oleh karena itu, diperlukan pendekatan berbasis data (data-driven) untuk membantu perusahaan memprediksi risiko turnover secara lebih akurat. Melalui pemanfaatan teknik data science dan machine learning, proyek ini dikembangkan untuk menganalisis pola perilaku karyawan dan membangun model prediksi yang mampu membantu manajemen dalam mengambil keputusan strategis yang lebih tepat, preventif, dan berkelanjutan.

Masalah yang dihadapi : 

Saat ini, perusahaan belum memiliki sistem yang mampu mengidentifikasi secara dini karyawan yang berpotensi melakukan resign, sehingga menyulitkan manajemen dalam mengambil tindakan preventif yang tepat.


Proyek ini bertujuan untuk:

- Mengidentifikasi faktor-faktor utama yang berpengaruh terhadap tingkat employee turnover.
- Mengembangkan model prediksi untuk mengklasifikasikan karyawan yang berpotensi melakukan resign.
- Menyusun rekomendasi kebijakan dan strategi HR berbasis data untuk membantu perusahaan mengurangi tingkat turnover.


Metodologi 
Tahapan yang dilakukan dalam proyek ini:
- Data Cleaning dan Preprocessing.
- Exploratory Data Analysis (EDA).
- Feature Engineering.
- Training dan evaluasi model machine learning.
- Hyperparameter tuning.
- Model interpretability (feature importance).

Pilihan Model Machine Learning yang digunakan :
- Logistic Regression
- Random Forest
- XGBoost

Evaluasi Model
Model dievaluasi menggunakan beberapa metrik:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Kesimpulan :

1. Exploratory Data Analysis (EDA)
Ditemukan pola bahwa number_project, average_monthly_hours, time_spend_company, dan tingkat kepuasan memiliki hubungan kuat dengan risiko turnover.
Karyawan dengan kepuasan rendah, beban kerja tinggi, dan masa kerja singkat cenderung memiliki potensi turnover lebih tinggi.

2. Modelling
Beberapa algoritma diuji seperti Logistic Regression, Decision Tree, Random Forest, dan XGBoost.
Model berbasis ensemble (Random Forest dan XGBoost) menunjukkan performa lebih tinggi dibanding model linear.

3. Hyperparameter Tuning
Hyperparameter tuning berhasil meningkatkan stabilitas dan performa model, terutama pada Random Forest dan XGBoost.
Overfitting dapat dikurangi, meskipun model kompleks masih menunjukkan perbedaan kecil antara data train dan test.

4. Pemilihan Model Terbaik
XGBoost (Tuned) dipilih sebagai model terbaik karena memiliki kombinasi akurasi, recall, F1-score, dan ROC-AUC tertinggi pada data test.
Turnover dipengaruhi kombinasi faktor number_project, average_monthly_hours, time_spend_company yang juga mempengaruhi kepuasan kerja.
Model ini memiliki kemampuan generalisasi yang baik sehingga layak digunakan untuk prediksi turnover di dunia nyata.

Rekomendasi Strategi HR Berdasarkan Hasil Analisis

1. Pengaturan Beban Kerja (number_project)

- Hindari pemberian proyek yang terlalu banyak atau terlalu sedikit.
- Gunakan dashboard untuk memantau distribusi beban kerja.

2. Retensi Berdasarkan Masa Kerja (time_spend_company)

- Fokus pada karyawan dengan masa kerja 3–5 tahun yang rawan resign.
- Sediakan career path yang jelas.
- Lakukan diskusi karier secara rutin.

3. Peningkatan Kepuasan Kerja (satisfaction_level)

- Lakukan survei kepuasan karyawan setiap kuartal.
- Prioritaskan perbaikan pada aspek yang paling banyak dikeluhkan.
- Terapkan program employee recognition.

4. Transparansi Evaluasi Kinerja (last_evaluation)

- Terapkan sistem feedback dua arah.
- Gunakan penilaian berbasis hasil dan proses kerja.
- Hindari kondisi high performer yang mengalami overwork.

5. Manajemen Jam Kerja (average_monthly_hours)

- Terapkan kebijakan lembur dengan persetujuan terlebih dahulu.
- Sediakan fleksibilitas kerja (WFH dan jam kerja fleksibel).
- Pantau jam kerja melalui sistem monitoring digital.

6. Intervensi Preventif untuk Karyawan Berisiko Tinggi

- Gunakan model prediksi untuk mendeteksi risiko resign.
- Lakukan stay interview secara berkala.
- Siapkan intervensi seperti pengurangan workload, mentoring, dan diskusi karier.

7. Review Kebijakan Promosi (promotion_last_5years)

- Tingkatkan transparansi kriteria promosi.
- Pastikan proses promosi adil dan berbasis kompetensi.

8. Strategi Kompensasi (salary)

- Lakukan benchmarking gaji secara berkala.
- Berikan insentif berbasis kinerja dan tunjangan yang kompetitif.

Modern Machine Learning App for Employee Turnover Analysis powered by XGBoost :


https://turnover-prediction-app.streamlit.app



