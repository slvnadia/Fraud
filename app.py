import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

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

# Definisi kolom dan indeks fitur
PRE_SELECTION_COLUMNS = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
    'newbalanceDest', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT',
    'type_TRANSFER', 'isDrained'
]
SELECTED_FEATURE_NAMES = [
    'amount', 'oldbalanceOrg', 'oldbalanceDest', 'newbalanceDest', 'isDrained'
]
if model and scaler:
    SELECTED_INDICES = [PRE_SELECTION_COLUMNS.index(col) for col in SELECTED_FEATURE_NAMES]
else:
    SELECTED_INDICES = []

# =============================================================================
# Fungsi Bantuan untuk Input Nominal
# =============================================================================
def format_number(number_string):
    """Membersihkan dan memformat string angka dengan pemisah titik."""
    # Hapus semua karakter non-digit
    cleaned_string = re.sub(r'[^\d]', '', number_string)
    if not cleaned_string:
        return "0"
    # Konversi ke integer dan format dengan pemisah titik
    return f"{int(cleaned_string):,}".replace(",", ".")

# =============================================================================
# Antarmuka Pengguna (UI) Streamlit
# =============================================================================
st.title("🕵️ Deteksi Penipuan Transaksi Digital")
st.write("Aplikasi ini menggunakan model Machine Learning untuk memprediksi potensi penipuan.")

# Inisialisasi session state untuk menyimpan nilai input
if 'amount_str' not in st.session_state:
    st.session_state.amount_str = "0"
if 'oldbalanceOrg_str' not in st.session_state:
    st.session_state.oldbalanceOrg_str = "0"
if 'newbalanceOrig_str' not in st.session_state:
    st.session_state.newbalanceOrig_str = "0"
if 'oldbalanceDest_str' not in st.session_state:
    st.session_state.oldbalanceDest_str = "0"
if 'newbalanceDest_str' not in st.session_state:
    st.session_state.newbalanceDest_str = "0"

with st.form("transaction_form"):
    st.header("Masukkan Detail Transaksi")
    
    col1, col2 = st.columns(2)
    with col1:
        type_transaction = st.selectbox("Tipe Transaksi", ('TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'))
        # Gunakan st.text_input untuk kontrol penuh
        st.session_state.amount_str = st.text_input("Jumlah Transaksi (Amount)", value=st.session_state.amount_str)
        st.session_state.oldbalanceOrg_str = st.text_input("Saldo Awal Pengirim (oldbalanceOrg)", value=st.session_state.oldbalanceOrg_str)
    
    with col2:
        st.session_state.newbalanceOrig_str = st.text_input("Saldo Akhir Pengirim (newbalanceOrig)", value=st.session_state.newbalanceOrig_str)
        st.session_state.oldbalanceDest_str = st.text_input("Saldo Awal Penerima (oldbalanceDest)", value=st.session_state.oldbalanceDest_str)
        st.session_state.newbalanceDest_str = st.text_input("Saldo Akhir Penerima (newbalanceDest)", value=st.session_state.newbalanceDest_str)

    submit_button = st.form_submit_button(label="Cek Transaksi")

# =============================================================================
# Logika Prediksi
# =============================================================================
if submit_button:
    if not all([model, scaler, SELECTED_INDICES]):
        st.error("Aplikasi tidak dapat melakukan prediksi karena model gagal dimuat.")
    else:
        with st.spinner('Memproses dan memprediksi...'):
            try:
                # 1. Bersihkan dan konversi input string ke float
                amount = float(re.sub(r'[^\d]', '', st.session_state.amount_str))
                oldbalanceOrg = float(re.sub(r'[^\d]', '', st.session_state.oldbalanceOrg_str))
                newbalanceOrig = float(re.sub(r'[^\d]', '', st.session_state.newbalanceOrig_str))
                oldbalanceDest = float(re.sub(r'[^\d]', '', st.session_state.oldbalanceDest_str))
                newbalanceDest = float(re.sub(r'[^\d]', '', st.session_state.newbalanceDest_str))

                # 2. Kumpulkan data untuk DataFrame
                data = {
                    'type': type_transaction, 'amount': amount, 'oldbalanceOrg': oldbalanceOrg,
                    'newbalanceOrig': newbalanceOrig, 'oldbalanceDest': oldbalanceDest, 'newbalanceDest': newbalanceDest
                }
                
                # 3. Lakukan pra-pemrosesan (sama seperti sebelumnya)
                input_df = pd.DataFrame([data])
                input_df['isDrained'] = (np.abs(input_df['oldbalanceOrg'] - input_df['amount']) < 0.01).astype(int)
                all_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
                input_df['type'] = pd.Categorical(input_df['type'], categories=all_types)
                input_df_encoded = pd.get_dummies(input_df, prefix='type', drop_first=True)
                input_df_aligned = input_df_encoded.reindex(columns=PRE_SELECTION_COLUMNS, fill_value=0)
                
                scaled_features = scaler.transform(input_df_aligned)
                final_features = scaled_features[:, SELECTED_INDICES]
                
                # 4. Lakukan Prediksi
                prediction_proba = model.predict_proba(final_features)[:, 1]
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

# Format ulang nilai di session state setelah form disubmit (agar tampilan rapi)
st.session_state.amount_str = format_number(st.session_state.amount_str)
st.session_state.oldbalanceOrg_str = format_number(st.session_state.oldbalanceOrg_str)
st.session_state.newbalanceOrig_str = format_number(st.session_state.newbalanceOrig_str)
st.session_state.oldbalanceDest_str = format_number(st.session_state.oldbalanceDest_str)
st.session_state.newbalanceDest_str = format_number(st.session_state.newbalanceDest_str)
