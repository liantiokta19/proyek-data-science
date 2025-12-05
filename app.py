import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Harga Mobil Australia",
    page_icon="🚗",
    layout="centered"
)

# --- FUNGSI LOAD MODEL (FINAL & AMAN) ---
@st.cache_resource
def load_model_resources():
    # 1. Dapatkan lokasi (path) absolut dari file app.py ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Gabungkan dengan nama file model
    model_path = os.path.join(current_dir, 'car_price_prediction_model.pkl')
    
    # 3. Cek keberadaan file (untuk debugging)
    if not os.path.exists(model_path):
        st.error(f"CRITICAL ERROR: File model tidak ditemukan di: {model_path}")
        st.write("Daftar file di folder ini:", os.listdir(current_dir))
        return None

    try:
        data = joblib.load(model_path)
        return data
    except Exception as e:
        st.error(f"Gagal memuat model. Error: {e}")
        return None

# --- MAIN APP ---

# Load Resources
resources = load_model_resources()

if resources is None:
    st.stop() # Hentikan aplikasi jika model gagal dimuat

# Ekstrak komponen model
model = resources['model']
encoders = resources['encoders']
scaler = resources['scaler']
feature_names = resources['feature_names']

# Judul Aplikasi
st.title("🚗 Prediksi Kategori Harga Kendaraan")
st.markdown("Aplikasi ini memprediksi apakah harga mobil tergolong **Murah**, **Sedang**, **Mahal**, atau **Sangat Mahal** berdasarkan spesifikasinya.")

# --- SIDEBAR INPUT ---
st.sidebar.header("Spesifikasi Mobil")

def get_options(col_name):
    """Mengambil opsi unik dari encoder yang tersimpan"""
    if col_name in encoders:
        return list(encoders[col_name].classes_)
    return []

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox('Brand', get_options('Brand'))
        model_car = st.selectbox('Model', get_options('Model'))
        year = st.number_input('Tahun Pembuatan', min_value=1990, max_value=2025, value=2020)
        transmission = st.selectbox('Transmisi', get_options('Transmission'))
        
    with col2:
        body_type = st.selectbox('Tipe Body', get_options('BodyType'))
        fuel_type = st.selectbox('Tipe Bahan Bakar', get_options('FuelType'))
        kilometres = st.number_input('Kilometer (km)', min_value=0, value=50000, step=1000)
        
        # Input tambahan (sesuaikan dengan fitur di notebook)
        cylinders = st.number_input('Jumlah Silinder', min_value=2, max_value=12, value=4)
        # Tambahkan input lain jika diperlukan oleh model
        
    submit = st.form_submit_button("🔍 Prediksi Harga")

if submit:
    # 1. Siapkan Data Input (sesuai urutan feature_names saat training)
    input_data = {col: [0] for col in feature_names} # Inisialisasi dengan 0
    
    # 2. Isi data dari form
    # Pastikan nama key di sini SAMA PERSIS dengan nama kolom di feature_names
    input_data['Brand'] = [brand]
    input_data['Model'] = [model_car]
    input_data['Year'] = [year]
    input_data['Transmission'] = [transmission]
    input_data['BodyType'] = [body_type]
    input_data['FuelType'] = [fuel_type]
    input_data['Kilometres'] = [kilometres]
    
    # Handle kolom khusus (sesuai preprocessing notebook)
    # Di notebook Anda melakukan cleaning '4 cyl' -> 4 (int).
    # Jadi jika model dilatih dengan data bersih, inputnya langsung angka.
    if 'CylindersinEngine' in feature_names:
        input_data['CylindersinEngine'] = [cylinders] # Asumsi di notebook sudah jadi angka
    
    # Buat DataFrame
    df_input = pd.DataFrame(input_data)
    
    # --- PREPROCESSING ---
    try:
        # Encode Kategorikal
        for col in df_input.columns:
            if col in encoders:
                # Handle unknown labels (jika ada input baru yang tidak dikenal model)
                val = df_input.at[0, col]
                if val in encoders[col].classes_:
                    df_input.at[0, col] = encoders[col].transform([val])[0]
                else:
                    # Fallback strategis: pakai nilai modus/pertama
                    df_input.at[0, col] = encoders[col].transform([encoders[col].classes_[0]])[0]

        # Scaling (Normalisasi)
        # Pastikan urutan kolom sama persis dengan saat fit scaler
        df_final = df_input[feature_names]
        X_scaled = scaler.transform(df_final)
        
        # --- PREDIKSI ---
        prediction_idx = model.predict(X_scaled)[0]
        
        # --- TAMPILKAN HASIL ---
        # Mapping manual hasil (karena LabelEncoder mengurutkan abjad)
        # Biasanya: 0=Mahal, 1=Murah, 2=Sangat Mahal, 3=Sedang (Urutan Abjad)
        # ATAU sesuai urutan di notebook: 0=Murah, 1=Sedang, 2=Mahal, 3=Sangat Mahal (jika pakai pd.cut codes)
        
        # Kita ambil aman dengan menampilkan langsung nama kelas dari model
        # Jika model klasifikasi, classes_ akan berisi label aslinya
        
        # Cek tipe prediksi
        final_label = str(prediction_idx) # Default
        
        # Jika model menyimpan classes (RandomForestClassifier punya ini)
        # Namun karena Anda menyimpan model yang sudah di-fit dengan y yang sudah di-encode,
        # outputnya adalah ANGKA (0, 1, 2, 3).
        
        # Logika Mapping dari Notebook Anda:
        # bins = [0, 20000, 40000, 60000, np.inf]
        # labels = ['Murah', 'Sedang', 'Mahal', 'Sangat Mahal']
        # pd.cut(..., labels=labels) -> Kemudian di-LabelEncoder-kan.
        
        # Urutan LabelEncoder (Abjad): 
        # 0: Mahal
        # 1: Murah
        # 2: Sangat Mahal
        # 3: Sedang
        
        label_mapping = {
            0: "Mahal (> 40k - 60k AUD)",
            1: "Murah (< 20k AUD)",
            2: "Sangat Mahal (> 60k AUD)",
            3: "Sedang (20k - 40k AUD)"
        }
        
        final_result = label_mapping.get(prediction_idx, f"Kategori {prediction_idx}")

        st.subheader("Hasil Prediksi:")
        
        if "Murah" in final_result:
            st.success(f"💰 {final_result}")
        elif "Sedang" in final_result:
            st.info(f"💵 {final_result}")
        elif "Mahal" in final_result:
            st.warning(f"💸 {final_result}")
        else:
            st.error(f"💎 {final_result}")
            
        # Probabilitas
        proba = model.predict_proba(X_scaled)
        st.write("---")
        st.write("Probabilitas per Kategori:")
        
        # Buat dataframe probabilitas dengan nama label yang benar
        proba_df = pd.DataFrame(proba, columns=[label_mapping[i] for i in range(4)])
        st.bar_chart(proba_df.T)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        st.warning("Detail Error (untuk developer): Pastikan jumlah dan nama fitur input sama persis dengan 'feature_names' di model.")
