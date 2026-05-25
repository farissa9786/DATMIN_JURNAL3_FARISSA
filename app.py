import streamlit as st
import pickle
import numpy as np

with open('model_naive_bayes_smote.pkl', 'rb') as f:
    nb_model = pickle.load(f)
with open('scaler_nb.pkl', 'rb') as f:
    scaler_nb = pickle.load(f)
with open('model_linear_regression.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('scaler_lr.pkl', 'rb') as f:
    scaler_lr = pickle.load(f)

st.set_page_config(page_title="RS Sehat Sentosa", page_icon="🏥", layout="wide")
st.sidebar.title("🏥 RS Sehat Sentosa")
st.sidebar.markdown("---")
halaman = st.sidebar.radio("Pilih Halaman:", [
    "🔬 Prediksi Glucose (Linear Regression)",
    "🩺 Klasifikasi Diabetes (Naive Bayes)"
])

if halaman == "🔬 Prediksi Glucose (Linear Regression)":
    st.title("🔬 Prediksi Kadar Glukosa Darah")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        age            = st.number_input("Usia (Age)", min_value=0, max_value=120, value=45)
        pregnancies    = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=2)
        bmi            = st.number_input("BMI (x100)", min_value=0, max_value=9999, value=2800)
        blood_pressure = st.number_input("Blood Pressure (x10)", min_value=0, max_value=9999, value=750)
        hba1c          = st.number_input("HbA1c (x10)", min_value=0, max_value=999, value=55)
        ldl            = st.number_input("LDL (x10)", min_value=0, max_value=9999, value=1100)
        hdl            = st.number_input("HDL (x10)", min_value=0, max_value=9999, value=500)
    with col2:
        triglycerides       = st.number_input("Triglycerides (x10)", min_value=0, max_value=9999, value=700)
        waist_circumference = st.number_input("Lingkar Pinggang (x10)", min_value=0.0, max_value=2000.0, value=900.0)
        hip_circumference   = st.number_input("Lingkar Pinggul (x10)", min_value=0, max_value=9999, value=1000)
        whr                 = st.number_input("WHR (x100)", min_value=0, max_value=999, value=90)
        family_history      = st.selectbox("Riwayat Keluarga Diabetes", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        diet_type           = st.selectbox("Tipe Diet", [0, 1, 2])
        hypertension        = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        medication_use      = st.selectbox("Penggunaan Obat", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
    st.markdown("---")
    if st.button("🔍 Prediksi Glucose", use_container_width=True):
        input_lr = np.array([[age, pregnancies, bmi, blood_pressure, hba1c, ldl, hdl,
                              triglycerides, waist_circumference, hip_circumference,
                              whr, family_history, diet_type, hypertension, medication_use]])
        hasil = lr_model.predict(scaler_lr.transform(input_lr))[0]
        st.metric("Prediksi Kadar Glucose", f"{hasil:.0f}")
        if hasil > 1260:
            st.error("⚠️ Kadar Glucose TINGGI — Risiko Diabetes")
        elif hasil > 1000:
            st.warning("⚠️ Kadar Glucose SEDANG — Perlu Pemantauan")
        else:
            st.success("✅ Kadar Glucose NORMAL")
        st.info("Hasil ini bersifat prediktif. Keputusan medis tetap berada pada dokter.")

elif halaman == "🩺 Klasifikasi Diabetes (Naive Bayes)":
    st.title("🩺 Klasifikasi Risiko Diabetes")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        age            = st.number_input("Usia (Age)", min_value=0, max_value=120, value=45)
        pregnancies    = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=2)
        bmi            = st.number_input("BMI (x100)", min_value=0, max_value=9999, value=2800)
        glucose        = st.number_input("Kadar Glucose (x10)", min_value=0, max_value=9999, value=1200)
        blood_pressure = st.number_input("Blood Pressure (x10)", min_value=0, max_value=9999, value=750)
        hba1c          = st.number_input("HbA1c (x10)", min_value=0, max_value=999, value=55)
        ldl            = st.number_input("LDL (x10)", min_value=0, max_value=9999, value=1100)
        hdl            = st.number_input("HDL (x10)", min_value=0, max_value=9999, value=500)
    with col2:
        triglycerides       = st.number_input("Triglycerides (x10)", min_value=0, max_value=9999, value=700)
        waist_circumference = st.number_input("Lingkar Pinggang (x10)", min_value=0.0, max_value=2000.0, value=900.0)
        hip_circumference   = st.number_input("Lingkar Pinggul (x10)", min_value=0, max_value=9999, value=1000)
        whr                 = st.number_input("WHR (x100)", min_value=0, max_value=999, value=90)
        family_history      = st.selectbox("Riwayat Keluarga Diabetes", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        diet_type           = st.selectbox("Tipe Diet", [0, 1, 2])
        hypertension        = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        medication_use      = st.selectbox("Penggunaan Obat", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
    st.markdown("---")
    if st.button("🔍 Klasifikasi Diabetes", use_container_width=True):
        input_nb = np.array([[age, pregnancies, bmi, glucose, blood_pressure, hba1c,
                              ldl, hdl, triglycerides, waist_circumference,
                              hip_circumference, whr, family_history, diet_type,
                              hypertension, medication_use]])
        input_scaled = scaler_nb.transform(input_nb)
        hasil = nb_model.predict(input_scaled)[0]
        proba = nb_model.predict_proba(input_scaled)[0]
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Hasil Klasifikasi", "DIABETES" if hasil == 1 else "NON-DIABETES")
        with col_r2:
            st.metric("Prob. Non-Diabetes", f"{proba[0]*100:.1f}%")
        with col_r3:
            st.metric("Prob. Diabetes", f"{proba[1]*100:.1f}%")
        if hasil == 1:
            st.error("🚨 Pasien TERDETEKSI DIABETES — Segera lakukan pemeriksaan lanjutan!")
        else:
            st.success("✅ Pasien TIDAK TERDETEKSI DIABETES — Tetap jaga pola hidup sehat.")
        st.info("Hasil ini bersifat prediktif. Keputusan medis tetap berada pada dokter.")
