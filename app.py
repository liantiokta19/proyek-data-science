import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Harga Mobil Australia",
    page_icon="🚗",
    layout="centered"
)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model_resources():
    try:
        # Memuat file model yang sudah disimpan dari Notebook
        data = joblib.load('car_price_prediction_model.pkl')
        return data
    except FileNotFoundError:
        return None

# Load Resources
resources = load_model_resources()

if resources is None:
    st.error("File 'car_price_prediction_model.pkl' tidak ditemukan. Silakan jalankan kode penyimpanan di Notebook dan pindahkan file .pkl ke folder ini.")
    st.stop()

model = resources['model']
encoders = resources['encoders']
scaler = resources['scaler']
feature_names = resources['feature_names']

# Judul
st.title("🚗 Prediksi Kategori Harga Kendaraan")
st.markdown("Aplikasi ini menggunakan model **Random Forest** yang telah dilatih (Pre-trained) untuk memprediksi kategori harga.")

# --- INPUT USER ---
st.sidebar.header("Masukkan Spesifikasi Mobil")

# Kita perlu mengambil opsi unik untuk Dropdown dari Encoders yang tersimpan
# Encoders adalah dictionary: {'Brand': LabelEncoderObject, ...}

def get_options(col_name):
    if col_name in encoders:
        return list(encoders[col_name].classes_)
    return []

# Form Input (Disesuaikan dengan fitur yang ada di X_train notebook Anda)
# Pastikan urutan dan nama fitur input ini nanti cocok dengan feature_names model

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox('Brand', get_options('Brand'))
        model_car = st.selectbox('Model', get_options('Model')) # Notebook pakai Model
        year = st.number_input('Year', min_value=1990, max_value=2024, value=2020)
        transmission = st.selectbox('Transmission', get_options('Transmission'))
        
    with col2:
        body_type = st.selectbox('Body Type', get_options('BodyType'))
        fuel_type = st.selectbox('Fuel Type', get_options('FuelType'))
        kilometres = st.number_input('Kilometres', min_value=0, value=50000)
        # Input numerik lain sesuai notebook
        cylinders = st.number_input('Cylinders', min_value=2, max_value=12, value=4)
        fuel_consumption = st.number_input('Fuel Consumption (L/100km)', min_value=0.0, value=8.0)
        engine_size = st.number_input('Engine Size (L)', min_value=0.0, value=2.0)
        
    # Input tambahan yang mungkin ada di model notebook (Set default jika user tidak perlu isi)
    # Karena di notebook X memiliki banyak kolom, kita harus membuat dummy data untuk kolom yang tidak ada di input form
    # agar bentuk array-nya sama.
    
    submit = st.form_submit_button("Prediksi Harga")

if submit:
    # 1. Siapkan Dictionary Data Input
    # Kita harus membuat DataFrame yang strukturnya SAMA PERSIS dengan X_train di notebook
    
    # Buat data awal dengan nilai default (0 atau mode) untuk semua fitur yang dilatih
    input_data = {col: [0] for col in feature_names}
    
    # Isi data dari input user
    input_data['Brand'] = [brand]
    input_data['Model'] = [model_car]
    input_data['Year'] = [year]
    input_data['Transmission'] = [transmission]
    input_data['BodyType'] = [body_type]
    input_data['FuelType'] = [fuel_type]
    input_data['Kilometres'] = [kilometres]
    input_data['CylindersinEngine'] = [f"{cylinders} cyl"] # Format harus sama dengan raw data sebelum cleaning di notebook jika encoder butuh raw
    # Atau jika di notebook sudah jadi angka:
    # input_data['CylindersinEngine'] = [cylinders] 
    
    # Catatan: Di notebook Anda melakukan cleaning '4 cyl' -> 4.
    # Karena kita memuat LabelEncoder dari notebook, kita harus hati-hati.
    # Jika LabelEncoder di-fit SETELAH cleaning (menjadi angka), maka input harus angka.
    # Jika LabelEncoder di-fit SEBELUM cleaning (masih string), input harus string.
    
    # Berdasarkan Notebook Anda: Cleaning dilakukan DULUAN, baru Label Encoding.
    # Jadi input ke model harus sudah bersih (angka).
    
    # Buat DataFrame sementara
    df_input = pd.DataFrame(input_data)
    
    # --- PREPROCESSING INPUT USER (Meniru Notebook) ---
    # Kita override nilai manual yang sudah bersih ke kolom dataframe
    df_input['Cylinders'] = cylinders # Fitur hasil cleaning di notebook
    df_input['FuelConsumption'] = fuel_consumption # Fitur hasil cleaning
    df_input['EngineSize'] = engine_size # Fitur hasil cleaning
    
    # Label Encoding untuk kolom Kategori
    # Loop semua kolom, jika ada di encoder, transform
    for col in df_input.columns:
        if col in encoders:
            try:
                # Ambil nilai input
                val = df_input.at[0, col]
                # Transform menggunakan encoder yang tersimpan
                df_input.at[0, col] = encoders[col].transform([val])[0]
            except Exception as e:
                # Jika nilai tidak dikenal (misal model mobil baru), pakai nilai default/pertama
                df_input.at[0, col] = 0
    
    # Scaling (StandardScaler)
    # Pastikan urutan kolom df_input SAMA dengan feature_names
    # Karena notebook Anda mungkin membuat kolom baru saat cleaning (seperti 'Cylinders' dari 'CylindersinEngine')
    # Kita harus menyusun ulang df_input agar hanya berisi kolom yang ada di feature_names
    
    # Filter hanya kolom yang digunakan model
    df_final = df_input[feature_names] 
    
    # Lakukan Scaling
    try:
        X_scaled = scaler.transform(df_final)
        
        # Prediksi
        prediction_idx = model.predict(X_scaled)[0]
        
        # Mapping hasil prediksi (0,1,2,3) kembali ke Label (Murah, Sedang, dll)
        # Di notebook: labels = ['Murah', 'Sedang', 'Mahal', 'Sangat Mahal']
        # Tapi LabelEncoder mengurutkan abjad: Mahal, Murah, Sangat Mahal, Sedang?
        # Kita perlu cek classes_ dari target di notebook. 
        # Asumsi sederhana berdasarkan binning manual Anda:
        
        # Kategori harga di notebook Anda menggunakan pd.cut dengan labels=['Murah', 'Sedang', 'Mahal', 'Sangat Mahal']
        # Hasil pd.cut adalah kategori berurut.
        
        st.success(f"Prediksi Kategori Harga: **{prediction_idx}**")
        
        # Tampilkan Probabilitas
        proba = model.predict_proba(X_scaled)
        st.write("Tingkat Keyakinan Model:")
        st.bar_chart(pd.DataFrame(proba.T, columns=["Probabilitas"], index=model.classes_))
        
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pemrosesan data: {e}")
        st.warning("Pastikan fitur yang dimasukkan di input sama dengan fitur yang digunakan saat training di Notebook.")
