import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =============================================================================
# Konfigurasi Awal dan Pemuatan Model
# =============================================================================
st.set_page_config(page_title="Deteksi Penipuan Transaksi", layout="centered")

@st.cache_resource
def load_artifacts():
    """Memuat model dan scaler dari file."""
    try:
        artifacts_path = 'model_artifacts'
        model = joblib.load(os.path.join(artifacts_path, 'fraud_detection_model.joblib'))
        scaler = joblib.load(os.path.join(artifacts_path, 'scaler.joblib'))
        return model, scaler
    except Exception as e:
        st.error(f"Error memuat artefak model: {e}")
        return None, None

model, scaler = load_artifacts()

# Definisi kolom sesuai dengan model LAMA (tanpa 'isDrained')
PRE_SELECTION_COLUMNS = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
    'newbalanceDest', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT',
    'type_TRANSFER'
]

# Definisikan 5 fitur yang dipilih oleh model LAMA
SELECTED_FEATURE_NAMES = [
    'amount', 'oldbalanceOrg', 'oldbalanceDest', 'newbalanceDest', 'type_TRANSFER'
]

# Dapatkan indeks dari fitur yang terpilih
if model and scaler:
    SELECTED_INDICES = [PRE_SELECTION_COLUMNS.index(col) for col in SELECTED_FEATURE_NAMES]
else:
    SELECTED_INDICES = []

# =============================================================================
# Antarmuka Pengguna (UI) Streamlit
# =============================================================================
st.title("🕵️ Deteksi Penipuan Transaksi Digital")
st.write("Aplikasi ini menggunakan model Machine Learning untuk memprediksi potensi penipuan.")

with st.form("transaction_form"):
    st.header("Masukkan Detail Transaksi")
    
    col1, col2 = st.columns(2)
    with col1:
        type_transaction = st.selectbox("Tipe Transaksi", ('TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'))
        # PERBAIKAN: Tambahkan value=None dan placeholder agar kolom kosong
        amount = st.number_input("Jumlah Transaksi (Amount)", min_value=0.0, format="%.2f", value=None, placeholder="Contoh: 500000")
        oldbalanceOrg = st.number_input("Saldo Awal Pengirim (oldbalanceOrg)", min_value=0.0, format="%.2f", value=None, placeholder="Contoh: 1000000")
    
    with col2:
        newbalanceOrig = st.number_input("Saldo Akhir Pengirim (newbalanceOrig)", min_value=0.0, format="%.2f", value=None, placeholder="Contoh: 500000")
        oldbalanceDest = st.number_input("Saldo Awal Penerima (oldbalanceDest)", min_value=0.0, format="%.2f", value=None, placeholder="Contoh: 200000")
        newbalanceDest = st.number_input("Saldo Akhir Penerima (newbalanceDest)", min_value=0.0, format="%.2f", value=None, placeholder="Contoh: 700000")

    submit_button = st.form_submit_button(label="Cek Transaksi")

# =============================================================================
# Logika Prediksi
# =============================================================================
if submit_button:
    # PERBAIKAN: Tambahkan validasi untuk memastikan semua kolom diisi
    if any(v is None for v in [amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]):
        st.warning("Harap isi semua kolom nominal sebelum melakukan prediksi.")
    elif not all([model, scaler, SELECTED_INDICES]):
        st.error("Aplikasi tidak dapat melakukan prediksi karena model gagal dimuat.")
    else:
        with st.spinner('Memproses dan memprediksi...'):
            try:
                # 1. Kumpulkan data input
                data = {
                    'type': type_transaction, 'amount': amount, 'oldbalanceOrg': oldbalanceOrg,
                    'newbalanceOrig': newbalanceOrig, 'oldbalanceDest': oldbalanceDest, 'newbalanceDest': newbalanceDest
                }
                
                # 2. Lakukan pra-pemrosesan sesuai model LAMA
                input_df = pd.DataFrame([data])
                
                all_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
                input_df['type'] = pd.Categorical(input_df['type'], categories=all_types)
                input_df_encoded = pd.get_dummies(input_df, prefix='type', drop_first=True)
                
                input_df_aligned = input_df_encoded.reindex(columns=PRE_SELECTION_COLUMNS, fill_value=0)
                
                # 3. Lakukan Scaling dan Feature Selection
                scaled_features = scaler.transform(input_df_aligned)
                final_features = scaled_features[:, SELECTED_INDICES]
                
                # 4. Lakukan Prediksi
                prediction_proba = model.predict_proba(final_features)[:, 1]
                # Model lama mungkin menggunakan threshold 0.9, kita kembalikan ke 0.5 untuk standar
                threshold = 0.5
                is_fraud = prediction_proba[0] >= threshold
                
                # 5. Tampilkan Hasil
                st.subheader("Hasil Prediksi:")
                formatted_amount = f"Rp {int(amount):,}".replace(",", ".")
                if is_fraud:
                    st.error(f"Transaksi Terdeteksi PENIPUAN (Probabilitas: {prediction_proba[0]*100:.2f}%)")
                    st.warning(f"Disarankan untuk tidak melanjutkan transaksi sebesar **{formatted_amount}** ini.")
                else:
                    st.success(f"Transaksi Dianggap AMAN (Probabilitas Penipuan: {prediction_proba[0]*100:.2f}%)")
                    st.info(f"Model memprediksi transaksi sebesar **{formatted_amount}** ini sebagai aktivitas wajar.")

            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")
