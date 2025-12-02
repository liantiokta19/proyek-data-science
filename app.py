import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Harga Mobil Australia",
    page_icon="🚗",
    layout="wide"
)

# Judul Aplikasi
st.title("🚗 Aplikasi Prediksi Kategori Harga Kendaraan")
st.markdown("Aplikasi ini menggunakan Machine Learning untuk memprediksi **Kategori Harga** kendaraan berdasarkan spesifikasinya.")

# --- 1. LOAD & PREPROCESS DATA ---
@st.cache_data
def load_and_clean_data():
    # Load Data
    try:
        df = pd.read_csv('Australian Vehicle Prices.csv')
    except FileNotFoundError:
        st.error("File 'Australian Vehicle Prices.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
        return None, None

    # Hapus data duplikat dan missing values sederhana
    df = df.dropna()
    df = df.drop_duplicates()

    # --- Cleaning Kolom Numerik yang bercampur string ---
    # Membersihkan 'Doors' (contoh: "4 Doors" -> 4)
    if df['Doors'].dtype == 'O':
        df['Doors'] = df['Doors'].str.extract('(\d+)').astype(float)
    
    # Membersihkan 'Seats'
    if df['Seats'].dtype == 'O':
        df['Seats'] = df['Seats'].str.extract('(\d+)').astype(float)
        
    # Membersihkan 'CylindersinEngine' (contoh: "4 cyl" -> 4)
    if df['CylindersinEngine'].dtype == 'O':
        df['CylindersinEngine'] = df['CylindersinEngine'].str.extract('(\d+)').astype(float)

    # Membersihkan 'FuelConsumption' (contoh: "8.7 L / 100 km" -> 8.7)
    if df['FuelConsumption'].dtype == 'O':
        df['FuelConsumption'] = df['FuelConsumption'].str.extract('(\d+\.?\d*)').astype(float)

    # Membersihkan 'Kilometres' (kadang ada koma atau tanda strip)
    if df['Kilometres'].dtype == 'O':
         df['Kilometres'] = pd.to_numeric(df['Kilometres'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

    # Pastikan Price numerik
    if df['Price'].dtype == 'O':
         df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

    df = df.dropna() # Drop lagi jika ada hasil parsing yang NaN

    # --- Membuat Kategori Harga (Target Variable) ---
    # Kita bagi harga menjadi 4 kategori: Budget, Standard, Premium, Luxury
    # Menggunakan qcut untuk pembagian yang merata berdasarkan kuartil
    df['Price_Category'] = pd.qcut(df['Price'], q=4, labels=['Budget', 'Standard', 'Premium', 'Luxury'])
    
    return df

df = load_and_clean_data()

if df is not None:
    # Tampilkan Data Sekilas
    with st.expander("🔍 Lihat Sampel Data Bersih"):
        st.dataframe(df.head())

    # --- 2. VISUALISASI DATA ---
    st.subheader("📊 Statistik Ringkas")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribusi Kategori Harga**")
        fig_count, ax_count = plt.subplots()
        sns.countplot(x='Price_Category', data=df, palette='viridis', ax=ax_count)
        st.pyplot(fig_count)

    with col2:
        st.markdown("**Top 10 Brand Kendaraan**")
        top_brands = df['Brand'].value_counts().head(10).index
        fig_brand, ax_brand = plt.subplots()
        sns.countplot(y='Brand', data=df[df['Brand'].isin(top_brands)], order=top_brands, palette='magma', ax=ax_brand)
        st.pyplot(fig_brand)

    # --- 3. TRAINING MODEL (Background) ---
    @st.cache_resource
    def train_model(data):
        # Fitur yang akan digunakan
        features = ['Brand', 'Year', 'Transmission', 'FuelType', 'Kilometres', 'Doors', 'Seats']
        target = 'Price_Category'
        
        X = data[features].copy()
        y = data[target]

        # Encoding Variabel Kategorikal
        encoders = {}
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        
        return model, encoders, acc

    model, encoders, accuracy = train_model(df)
    
    st.success(f"Model berhasil dilatih dengan Akurasi: **{accuracy:.2%}**")

    # --- 4. INPUT USER UNTUK PREDIKSI ---
    st.sidebar.header("📝 Masukkan Spesifikasi Mobil")

    def user_input_features():
        brand = st.sidebar.selectbox('Brand', df['Brand'].unique())
        year = st.sidebar.slider('Year', int(df['Year'].min()), int(df['Year'].max()), 2018)
        transmission = st.sidebar.selectbox('Transmission', df['Transmission'].unique())
        fuel_type = st.sidebar.selectbox('Fuel Type', df['FuelType'].unique())
        
        kilometres = st.sidebar.number_input('Kilometres', min_value=0, value=50000)
        doors = st.sidebar.slider('Doors', 2, 5, 4)
        seats = st.sidebar.slider('Seats', 2, 8, 5)

        data = {
            'Brand': brand,
            'Year': year,
            'Transmission': transmission,
            'FuelType': fuel_type,
            'Kilometres': kilometres,
            'Doors': doors,
            'Seats': seats
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    st.subheader("🔮 Hasil Prediksi")
    st.write("Spesifikasi yang Anda masukkan:")
    st.dataframe(input_df)

    if st.button("Prediksi Harga"):
        # Preprocess input user sama seperti training data
        input_processed = input_df.copy()
        
        for col, le in encoders.items():
            # Handle unknown labels (jika user input sesuatu yang tidak ada di training data)
            # Karena kita pakai selectbox dari unique values, resiko ini kecil, tapi tetap perlu dijaga
            try:
                input_processed[col] = le.transform(input_processed[col].astype(str))
            except ValueError:
                # Fallback strategis: assign ke label yang paling umum atau 0
                input_processed[col] = 0 
        
        prediction = model.predict(input_processed)[0]
        
        # Tampilkan Hasil
        if prediction == 'Budget':
            st.info(f"Kategori Harga: **{prediction}** (Sangat Terjangkau)")
        elif prediction == 'Standard':
            st.success(f"Kategori Harga: **{prediction}** (Menengah)")
        elif prediction == 'Premium':
            st.warning(f"Kategori Harga: **{prediction}** (Mahal)")
        else:
            st.error(f"Kategori Harga: **{prediction}** (Mewah/Luxury)")
            
        # Tampilkan probabilitas
        proba = model.predict_proba(input_processed)
        st.write("Probabilitas per kelas:")
        proba_df = pd.DataFrame(proba, columns=model.classes_)
        st.bar_chart(proba_df.T)
else:
    st.warning("Data belum dimuat.")
