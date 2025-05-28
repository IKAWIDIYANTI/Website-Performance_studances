import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr

# Set page config
st.set_page_config(
    page_title="Prediksi Risiko Dropout Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul aplikasi
st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.markdown("""
Aplikasi ini membantu memprediksi risiko *dropout* mahasiswa berdasarkan berbagai faktor demografi, akademik, dan ekonomi.
Tujuannya adalah untuk mengidentifikasi mahasiswa berisiko tinggi sejak dini dan memberikan dukungan yang tepat.
""")

# Fungsi untuk memuat aset prediksi
@st.cache_resource(ttl=3600)
def load_prediction_assets():
    """Memuat model, scaler, frequency maps, dan daftar fitur yang sudah disimpan."""
    try:
        model = joblib.load('model/xgboost_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        high_card_freq_maps = joblib.load('model/high_card_freq.pkl')
        model_features = joblib.load('model/model_features.pkl')
        return model, scaler, high_card_freq_maps, model_features
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, None, None

# Memuat aset
model, scaler, high_card_freq_maps, model_features = load_prediction_assets()

# Jika model tidak berhasil dimuat, hentikan aplikasi
if model is None:
    st.error("Gagal memuat model. Pastikan file model sudah ada di direktori 'model/'. Aplikasi tidak dapat berjalan.")
    st.stop()

# Fungsi untuk melakukan preprocessing input
def preprocess_input(input_df, scaler, high_card_freq_maps, model_features):
    """
    Melakukan preprocessing pada input dari pengguna agar sesuai dengan format model.
    """
    processed_df = input_df.copy()

    high_cardinality = ['Application_mode', 'Course', 'Previous_qualification', 'Nacionality',
                        'Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation']

    for col in high_cardinality:
        if col in processed_df.columns:
            # Gunakan frequency mapping
            processed_df[col + '_freq'] = processed_df[col].map(high_card_freq_maps.get(col, {})).fillna(0)
            processed_df = processed_df.drop(col, axis=1)

    # One-hot encoding untuk Marital_status
    if 'Marital_status' in processed_df.columns:
        # Dummy coding, tapi kita akan pastikan semua kategori yang ada di training ada di sini
        processed_df = pd.get_dummies(processed_df, columns=['Marital_status'], prefix='Marital_status', drop_first=False)

    # Buat DataFrame dengan semua kolom yang diharapkan model, diisi dengan 0
    final_df = pd.DataFrame(0, index=processed_df.index, columns=model_features)
    
    # Salin nilai dari processed_df ke final_df untuk kolom yang ada
    for col in processed_df.columns:
        if col in final_df.columns:
            final_df[col] = processed_df[col]
        # Jika kolom ada di processed_df tapi tidak di model_features, abaikan (karena mungkin tidak diperlukan model)

    # Kolom numerik yang perlu di-scale
    numerical_cols = ['Previous_qualification_grade', 'Admission_grade', 'Age_at_enrollment',
                      'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
                      'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
                      'Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_without_evaluations',
                      'Unemployment_rate', 'Inflation_rate', 'GDP']
    
    # Hanya kolom yang ada di final_df dan di numerical_cols
    numerical_cols_to_scale = [col for col in numerical_cols if col in final_df.columns]
    
    if numerical_cols_to_scale:
        final_df[numerical_cols_to_scale] = scaler.transform(final_df[numerical_cols_to_scale])

    return final_df

# Sidebar untuk input
st.sidebar.header("Input Data Mahasiswa")
st.sidebar.markdown("Isi detail berikut untuk memprediksi risiko *dropout*.")

# Mapping untuk pilihan kategorikal
marital_status_map = {
    1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto Union", 6: "Legally Separated"
}
application_mode_map = {
    1: "1st phase - general contingent", 2: "Order re-enrolment", 3: "Change of course",
    5: "1st phase - special contingent (Azores Island)", 7: "Change of institution",
    10: "1st phase - special contingent (Madeira Island)", 15: "2nd phase - general contingent",
    17: "3rd phase - general contingent", 26: "Institutional transfer", 27: "Student with international higher education",
    39: "1st phase - special contingent (scholarship holder from other institution)", 42: "2nd phase - special contingent (scholarship holder from other institution)"
}
course_map = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (daytime)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service (evening)", 9254: "Tourism",
    9500: "Nursing", 9556: "Oral Hygiene", 9670: "Veterinary Medicine",
    9773: "Environmental Health", 9853: "Nutrition", 9991: "European Global Studies",
    9992: "Sport and Physical Education", 9993: "Public Administration"
}
qualification_map = {
    1: "Secondary Education", 2: "Higher Education - bachelor's degree",
    3: "Higher Education - degree", 4: "Higher Education - master's",
    5: "Higher Education - doctorate", 6: "Lived outside Portugal", 9: "Technological specialization course",
    10: "Higher Education - Doutoramento", 11: "Higher Education - Licenciatura",
    12: "Higher Education - Mestrado", 14: "Other", 18: "11th year of schooling",
    19: "12th year of schooling", 20: "9th year of schooling", 21: "Higher Education - bacharelato",
    22: "Technological education, specialized training, and/or professional training courses",
    25: "General secondary course", 29: "Basic Education 3rd Cycle (9th/10th/11th Year) or equiv.",
    30: "Basic Education 2nd Cycle (6th Year) or equiv.", 31: "Basic Education 1st Cycle (4th Year) or equiv.",
    32: "No education - (0 to 4 years)", 33: "Higher Education - other",
    34: "Basic Education 3rd Cycle (9th Year) or equiv.", 35: "Basic Education 2nd Cycle (6th Year) or equiv.",
    36: "Basic Education 1st Cycle (4th Year) or equiv.", 37: "Other (non-specified)",
    38: "Secondary Education (Technical or Professional Course)", 39: "Higher Education - Master's (non-specified)"
}
occupation_map = {
    0: "Student", 1: "Representatives of the Legislative Power",
    2: "Specialists of the Intellectual and Scientific Activities", 3: "Intermediate Level Technicians",
    4: "Administrative Staff", 5: "Personal Services, Security and Surveillance, and Sellers",
    6: "Farmers and Skilled Workers in Agriculture, Fisheries, and Forestry",
    7: "Skilled Workers in Industry, Construction, and Craftsmen", 8: "Installation and Machine Operators and Assembly Workers",
    9: "Unskilled Workers", 10: "Armed Forces Professionals", 11: "Other",
    12: "Unemployed", 13: "Administrative / Office Assistant", 14: "Sales Assistant",
    15: "Teacher", 16: "Doctor", 17: "Engineer", 18: "Accountant", 19: "Manager",
    20: "Architect", 21: "Programmer", 22: "Nurse", 23: "Lawyer", 24: "Journalist",
    25: "Police Officer", 26: "Firefighter", 27: "Artist", 28: "Athlete", 29: "Chef",
    30: "Mechanic", 31: "Electrician", 32: "Plumber", 33: "Construction Worker",
    34: "Cleaner", 35: "Driver", 36: "Security Guard", 37: "Retired", 38: "Homemaker",
    39: "Self-employed"
}

# Input data
marital_status = st.sidebar.selectbox("Status Pernikahan", options=list(marital_status_map.keys()), format_func=lambda x: marital_status_map[x])
application_mode = st.sidebar.selectbox("Mode Aplikasi", options=list(application_mode_map.keys()), format_func=lambda x: application_mode_map[x])
course = st.sidebar.selectbox("Pilihan Jurusan", options=list(course_map.keys()), format_func=lambda x: course_map[x])
daytime_evening_attendance = st.sidebar.selectbox("Kehadiran Siang/Malam", options=[1, 0], format_func=lambda x: "Siang" if x == 1 else "Malam")
previous_qualification = st.sidebar.number_input("Kualifikasi Sebelumnya (Kode)", min_value=1, value=1)
nacionality = st.sidebar.number_input("Kewarganegaraan (Kode)", min_value=1, value=1)
mothers_qualification = st.sidebar.selectbox("Kualifikasi Pendidikan Ibu", options=list(qualification_map.keys()), format_func=lambda x: qualification_map[x])
fathers_qualification = st.sidebar.selectbox("Kualifikasi Pendidikan Ayah", options=list(qualification_map.keys()), format_func=lambda x: qualification_map[x])
mothers_occupation = st.sidebar.selectbox("Pekerjaan Ibu", options=list(occupation_map.keys()), format_func=lambda x: occupation_map[x])
fathers_occupation = st.sidebar.selectbox("Pekerjaan Ayah", options=list(occupation_map.keys()), format_func=lambda x: occupation_map[x])
application_order = st.sidebar.number_input("Urutan Aplikasi (misal: 1 jika pertama)", min_value=1, value=1)
previous_qualification_grade = st.sidebar.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, value=120.0)
admission_grade = st.sidebar.number_input("Nilai Kelulusan (Admission Grade)", min_value=0.0, value=140.0)
age_at_enrollment = st.sidebar.number_input("Usia Saat Pendaftaran", min_value=17, value=20)
debtor = st.sidebar.selectbox("Status Debitor?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
tuition_fees_up_to_date = st.sidebar.selectbox("Pembayaran SPP Tepat Waktu?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
gender = st.sidebar.selectbox("Jenis Kelamin", options=[1, 0], format_func=lambda x: "Pria" if x == 1 else "Wanita")
scholarship_holder = st.sidebar.selectbox("Penerima Beasiswa?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
international = st.sidebar.selectbox("Mahasiswa Internasional?", options=[0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
curricular_units_1st_sem_credited = st.sidebar.number_input("Jumlah SKS Dikecualikan Sem. 1", min_value=0, value=0)
curricular_units_1st_sem_enrolled = st.sidebar.number_input("Jumlah SKS Diambil Sem. 1", min_value=0, value=6)
curricular_units_1st_sem_evaluations = st.sidebar.number_input("Jumlah Ujian Sem. 1", min_value=0, value=6)
curricular_units_1st_sem_approved = st.sidebar.number_input("Jumlah SKS Disetujui Sem. 1", min_value=0, value=6)
curricular_units_1st_sem_grade = st.sidebar.number_input("Rata-rata Nilai Sem. 1", min_value=0.0, value=12.0)
curricular_units_1st_sem_without_evaluations = st.sidebar.number_input("Jumlah Mata Kuliah Tanpa Nilai Sem. 1", min_value=0, value=0)
unemployment_rate = st.sidebar.number_input("Tingkat Pengangguran (%)", min_value=0.0, value=10.0)
inflation_rate = st.sidebar.number_input("Tingkat Inflasi (%)", min_value=-5.0, value=2.0)
gdp = st.sidebar.number_input("Produk Domestik Bruto (GDP)", min_value=-10.0, value=1.0)

# Tombol prediksi
if st.sidebar.button("Prediksi Risiko Dropout"):
    input_data = pd.DataFrame({
        'Marital_status': [marital_status],
        'Application_mode': [application_mode],
        'Application_order': [application_order],
        'Course': [course],
        'Daytime_evening_attendance': [daytime_evening_attendance],
        'Previous_qualification': [previous_qualification],
        'Previous_qualification_grade': [previous_qualification_grade],
        'Nacionality': [nacionality],
        'Mothers_qualification': [mothers_qualification],
        'Fathers_qualification': [fathers_qualification],
        'Mothers_occupation': [mothers_occupation],
        'Fathers_occupation': [fathers_occupation],
        'Admission_grade': [admission_grade],
        'Age_at_enrollment': [age_at_enrollment],
        'Debtor': [debtor],
        'Tuition_fees_up_to_date': [tuition_fees_up_to_date],
        'Gender': [gender],
        'Scholarship_holder': [scholarship_holder],
        'International': [international],
        'Curricular_units_1st_sem_credited': [curricular_units_1st_sem_credited],
        'Curricular_units_1st_sem_enrolled': [curricular_units_1st_sem_enrolled],
        'Curricular_units_1st_sem_evaluations': [curricular_units_1st_sem_evaluations],
        'Curricular_units_1st_sem_approved': [curricular_units_1st_sem_approved],
        'Curricular_units_1st_sem_grade': [curricular_units_1st_sem_grade],
        'Curricular_units_1st_sem_without_evaluations': [curricular_units_1st_sem_without_evaluations],
        'Unemployment_rate': [unemployment_rate],
        'Inflation_rate': [inflation_rate],
        'GDP': [gdp]
    })
    
    # Preprocess input
    processed_input = preprocess_input(input_data, scaler, high_card_freq_maps, model_features)
    
    # Prediksi
    prediction_proba = model.predict_proba(processed_input)[:, 1][0]
    
    # Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    if prediction_proba > 0.5:
        st.error(f"⚠️ Risiko Dropout Tinggi: **{prediction_proba*100:.2f}%**")
        st.write("Mahasiswa ini berisiko tinggi untuk *dropout*. Pertimbangkan untuk memberikan dukungan akademik atau konseling.")
    else:
        st.success(f"✅ Risiko Dropout Rendah: **{prediction_proba*100:.2f}%**")
        st.write("Mahasiswa ini memiliki risiko *dropout* yang rendah. Terus pantau perkembangannya.")
    
    st.markdown("---")
    st.subheader("Detail Prediksi (Probabilitas):")
    st.metric(label="Probabilitas Dropout", value=f"{prediction_proba*100:.2f}%")

# Bagian analisis data
st.subheader("Analisis Data dari Dataset Asli")
st.markdown("""
Bagian ini menampilkan hasil analisis dari dataset asli untuk menjawab pertanyaan Anda.
""")

@st.cache_data
def run_analysis():
    df_path = 'data/data.csv'
    try:
        df_analysis = pd.read_csv(df_path, sep=';')
    except:
        st.error(f"File data tidak ditemukan: {df_path}")
        return None
    
    # Bersihkan nama kolom
    df_analysis.columns = df_analysis.columns.str.strip()
    
    # Normalisasi nama kolom target
    if 'status' in df_analysis.columns and 'Status' not in df_analysis.columns:
        df_analysis.rename(columns={'status': 'Status'}, inplace=True)
    elif 'STATUS' in df_analysis.columns and 'Status' not in df_analysis.columns:
        df_analysis.rename(columns={'STATUS': 'Status'}, inplace=True)
    
    if 'Status' not in df_analysis.columns:
        st.error("Kolom 'Status' tidak ditemukan.")
        return None
        
    # Buat target
    df_analysis['target'] = df_analysis['Status'].apply(lambda x: 1 if x == 'Dropout' else 0)
    return df_analysis

df_analysis = run_analysis()

if df_analysis is not None:
    # 1. Analisis status ekonomi
    st.subheader("1. Pengaruh Status Ekonomi terhadap Risiko Dropout")
    
    # Uji Chi-square untuk Debtor
    if 'Debtor' in df_analysis.columns:
        contingency_table_debtor = pd.crosstab(df_analysis['Debtor'], df_analysis['target'])
        chi2_debtor, p_debtor, _, _ = chi2_contingency(contingency_table_debtor)
        st.write(f"- **Status Debitor (Debtor):** P-value: {p_debtor:.4f} {'(Signifikan)' if p_debtor < 0.05 else '(Tidak Signifikan)'}")
        if p_debtor < 0.05:
            st.info("Status debitor memiliki pengaruh signifikan terhadap keputusan *dropout*.")
            st.dataframe(contingency_table_debtor)
        else:
            st.info("Status debitor tidak memiliki pengaruh signifikan terhadap keputusan *dropout*.")
    else:
        st.warning("Kolom 'Debtor' tidak ditemukan untuk analisis.")
    
    # Uji Chi-square untuk Tuition_fees_up_to_date
    if 'Tuition_fees_up_to_date' in df_analysis.columns:
        contingency_table_tuition = pd.crosstab(df_analysis['Tuition_fees_up_to_date'], df_analysis['target'])
        chi2_tuition, p_tuition, _, _ = chi2_contingency(contingency_table_tuition)
        st.write(f"- **Pembayaran SPP Tepat Waktu (Tuition_fees_up_to_date):** P-value: {p_tuition:.4f} {'(Signifikan)' if p_tuition < 0.05 else '(Tidak Signifikan)'}")
        if p_tuition < 0.05:
            st.info("Keterlambatan pembayaran SPP memiliki pengaruh signifikan terhadap keputusan *dropout*.")
            st.dataframe(contingency_table_tuition)
        else:
            st.info("Keterlambatan pembayaran SPP tidak memiliki pengaruh signifikan terhadap keputusan *dropout*.")
    else:
        st.warning("Kolom 'Tuition_fees_up_to_date' tidak ditemukan untuk analisis.")
    
    # 2. Analisis performa akademik
    st.subheader("2. Hubungan Performa Akademik Semester Pertama dengan Status Mahasiswa")
    
    if 'Curricular_units_1st_sem_grade' in df_analysis.columns:
        mean_grade_dropout = df_analysis[df_analysis['target']==1]['Curricular_units_1st_sem_grade'].mean()
        mean_grade_non_dropout = df_analysis[df_analysis['target']==0]['Curricular_units_1st_sem_grade'].mean()
        grade_diff_percent = ((mean_grade_non_dropout - mean_grade_dropout)/mean_grade_non_dropout)*100

        st.write(f"- **Rata-rata Nilai Semester Pertama:**")
        st.write(f"     - Mahasiswa Dropout: {mean_grade_dropout:.2f}")
        st.write(f"     - Mahasiswa Tidak Dropout: {mean_grade_non_dropout:.2f}")
        st.info(f"Rata-rata nilai mahasiswa yang *dropout* lebih rendah sekitar **{grade_diff_percent:.1f}%** dibanding yang tidak *dropout*.")
        
        # Visualisasi
        fig_grade, ax_grade = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='target', y='Curricular_units_1st_sem_grade', data=df_analysis, ax=ax_grade)
        ax_grade.set_title('Distribusi Nilai Semester Pertama vs Status Dropout')
        ax_grade.set_xlabel('Status Dropout (0 = Tidak, 1 = Ya)')
        ax_grade.set_ylabel('Nilai Semester Pertama')
        st.pyplot(fig_grade)
    else:
        st.warning("Kolom 'Curricular_units_1st_sem_grade' tidak ditemukan untuk analisis.")
    
    if 'Curricular_units_1st_sem_approved' in df_analysis.columns:
        mean_approved_dropout = df_analysis[df_analysis['target']==1]['Curricular_units_1st_sem_approved'].mean()
        mean_approved_non_dropout = df_analysis[df_analysis['target']==0]['Curricular_units_1st_sem_approved'].mean()

        st.write(f"- **Rata-rata Mata Kuliah Disetujui Semester Pertama:**")
        st.write(f"     - Mahasiswa Dropout: {mean_approved_dropout:.2f}")
        st.write(f"     - Mahasiswa Tidak Dropout: {mean_approved_non_dropout:.2f}")
        st.info(f"Mahasiswa yang *dropout* cenderung memiliki lebih sedikit mata kuliah yang disetujui di semester pertama.")
    else:
        st.warning("Kolom 'Curricular_units_1st_sem_approved' tidak ditemukan untuk analisis.")
    
    # 3. Analisis pendidikan orang tua
    st.subheader("3. Pengaruh Latar Belakang Pendidikan Orang Tua terhadap Keberhasilan Akademik")
    
    if 'Mothers_qualification' in df_analysis.columns and 'Curricular_units_1st_sem_grade' in df_analysis.columns:
        corr_mother, p_mother = spearmanr(df_analysis['Mothers_qualification'], df_analysis['Curricular_units_1st_sem_grade'])
        st.write(f"- **Korelasi Pendidikan Ibu dengan Nilai Semester Pertama (Spearman):** {corr_mother:.2f} (p-value: {p_mother:.4f})")
    else:
        st.warning("Kolom 'Mothers_qualification' atau 'Curricular_units_1st_sem_grade' tidak ditemukan untuk analisis.")
    
    if 'Fathers_qualification' in df_analysis.columns and 'Curricular_units_1st_sem_grade' in df_analysis.columns:
        corr_father, p_father = spearmanr(df_analysis['Fathers_qualification'], df_analysis['Curricular_units_1st_sem_grade'])
        st.write(f"- **Korelasi Pendidikan Ayah dengan Nilai Semester Pertama (Spearman):** {corr_father:.2f} (p-value: {p_father:.4f})")
    else:
        st.warning("Kolom 'Fathers_qualification' atau 'Curricular_units_1st_sem_grade' tidak ditemukan untuk analisis.")
    
    if 'p_mother' in locals() and 'p_father' in locals():
        if p_mother < 0.05 or p_father < 0.05:
            st.info("Terdapat korelasi yang signifikan antara tingkat pendidikan orang tua dengan performa akademik semester pertama mahasiswa.")
        else:
            st.info("Korelasi antara tingkat pendidikan orang tua dengan performa akademik semester pertama mahasiswa tidak signifikan.")
    
    # Visualisasi pendidikan orang tua vs dropout
    if 'Mothers_qualification' in df_analysis.columns and 'Fathers_qualification' in df_analysis.columns:
        fig_qual, axes_qual = plt.subplots(1, 2, figsize=(16, 6))
        
        if 'Mothers_qualification' in df_analysis.columns:
            sns.countplot(x='Mothers_qualification', hue='target', data=df_analysis, palette='viridis',
                         ax=axes_qual[0])
            axes_qual[0].set_title('Distribusi Pendidikan Ibu vs Status Dropout')
            axes_qual[0].set_xlabel('Tingkat Pendidikan Ibu (Kode Kategori)')
            axes_qual[0].set_ylabel('Jumlah Mahasiswa')
            axes_qual[0].legend(title='Dropout', labels=['Tidak', 'Ya'])
        
        if 'Fathers_qualification' in df_analysis.columns:
            sns.countplot(x='Fathers_qualification', hue='target', data=df_analysis, palette='viridis',
                         ax=axes_qual[1])
            axes_qual[1].set_title('Distribusi Pendidikan Ayah vs Status Dropout')
            axes_qual[1].set_xlabel('Tingkat Pendidikan Ayah (Kode Kategori)')
            axes_qual[1].set_ylabel('')
        
        plt.tight_layout()
        st.pyplot(fig_qual)
    else:
        st.warning("Kolom pendidikan orang tua tidak ditemukan untuk visualisasi.")

# Footer
st.markdown("---")
st.write("Dibuat dengan ❤️ oleh Jaya Jaya Institut untuk tujuan edukasi.")