import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Load model
model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Prediksi", "Data & Grafik", "Tentang"])

# =======================
# HALAMAN: PREDIKSI
# =======================
if menu == "Prediksi":
    st.title("🚗 Prediksi Harga Mobil")

    st.markdown("Masukkan informasi mobil di bawah ini untuk memprediksi harga jualnya.")

    # Tata letak kolom
    col1, col2 = st.columns(2)

    with col1:
        highwaympg = st.number_input("Highway MPG", min_value=10, max_value=60, step=1)
        horsepower = st.number_input("Horsepower", min_value=50, max_value=500, step=5)
    with col2:
        curbweight = st.number_input("Curb Weight (kg)", min_value=500, max_value=3000, step=50)
        carwidth = st.number_input("Car Width (cm)", min_value=100, max_value=300, step=1)

    if st.button("🔍 Prediksi Harga"):
        input_data = np.array([[highwaympg, curbweight, horsepower, carwidth]])
        car_prediction = model.predict(input_data)

        harga_mobil_float = float(car_prediction[0])
        harga_mobil_formatted = f"Rp {harga_mobil_float:,.2f}"

        st.success(f"💰 Harga Mobil yang Diprediksi: **{harga_mobil_formatted}**")

        # Tampilkan grafik perbandingan input dengan rata-rata dataset
        st.subheader("📊 Perbandingan Fitur dengan Rata-rata")
        df_avg = {
            'Fitur': ['Highway MPG', 'Curb Weight', 'Horsepower', 'Car Width'],
            'Input': [highwaympg, curbweight, horsepower, carwidth],
            'Rata-rata': [30, 1500, 150, 170]  # Sesuaikan dengan dataset asli
        }
        df_avg = pd.DataFrame(df_avg)

        chart = alt.Chart(df_avg).transform_fold(
            ['Input', 'Rata-rata'],
            as_=['Jenis', 'Nilai']
        ).mark_bar().encode(
            x='Fitur:N',
            y='Nilai:Q',
            color='Jenis:N'
        )
        st.altair_chart(chart, use_container_width=True)

# =======================
# HALAMAN: DATASET & GRAFIK
# =======================
elif menu == "Data & Grafik":
    st.title("📈 Eksplorasi Dataset Mobil")
    df = pd.read_csv("CarPrice.csv")
    st.dataframe(df.head())

    with st.expander("Lihat Grafik Interaktif"):
        st.write("🔸 Highway-mpg vs Harga")
        chart1 = alt.Chart(df).mark_circle(size=60).encode(
            x='highwaympg',
            y='price',
            tooltip=['highwaympg', 'price']
        ).interactive()
        st.altair_chart(chart1, use_container_width=True)

        st.write("🔸 Horsepower vs Harga")
        chart2 = alt.Chart(df).mark_circle(size=60, color="orange").encode(
            x='horsepower',
            y='price',
            tooltip=['horsepower', 'price']
        ).interactive()
        st.altair_chart(chart2, use_container_width=True)

# =======================
# HALAMAN: TENTANG
# =======================
elif menu == "Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dikembangkan menggunakan **Streamlit** dan **Scikit-Learn** untuk memprediksi harga mobil berdasarkan fitur-fitur penting seperti:
    
    - Highway MPG
    - Curb Weight
    - Horsepower
    - Car Width

    Model dilatih menggunakan data historis dari dataset `CarPrice.csv`.

    **Dibuat oleh:** Aditya Dimas Saputra  
    **Mata Kuliah:** Praktikum Kecerdasan Buatan  
    """)

    st.image("https://img.icons8.com/color/480/car--v1.png", width=150)
