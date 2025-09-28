# brain_dashboard_plotly.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objs as go
import xgboost as xgb

# -------------------------
# 1️⃣ Model Yükle
# -------------------------
best_model = joblib.load("best_xgb_model.pkl")  # XGBoost modeli

# -------------------------
# 2️⃣ Dashboard Başlığı ve Layout
# -------------------------
st.set_page_config(page_title="EEG Epilepsi Dashboard", layout="wide")
st.title("🧠 EEG Epilepsi Nöbeti Tespiti Dashboard (Interaktif)")

st.markdown("""
Bu dashboard XGBoost kullanarak EEG verisinden epilepsi nöbetini tahmin eder ve interaktif grafiklerle kriz bölgelerini gösterir.
""")

# -------------------------
# 3️⃣ Sidebar Kontrolleri
# -------------------------
st.sidebar.header("Kontroller")
uploaded_file = st.sidebar.file_uploader("EEG CSV dosyasını yükleyin", type=["csv"])
show_raw = st.sidebar.checkbox("Veriyi göster", value=True)
channel_select = st.sidebar.multiselect("Kanalları seç", options=None)
ma_window = st.sidebar.slider("Moving Average penceresi", 1, 50, 5)
highlight_crisis = st.sidebar.checkbox("Kriz bölgelerini göster", value=True)

# -------------------------
# 4️⃣ EEG Veri İşleme
# -------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    if show_raw:
        st.subheader("Yüklenen Veri")
        st.dataframe(data.head(5))
    
    # Kanal seçimi
    if not channel_select:
        channels = data.columns.tolist()
    else:
        channels = channel_select
    
    # Moving Average filtresi
    data_ma = data[channels].rolling(window=ma_window, min_periods=1).mean()
    
    # Model için ilk satır
    sample_row = pd.to_numeric(data_ma.iloc[0], errors='coerce').fillna(0).values
    n_features = best_model.n_features_in_
    if sample_row.shape[0] > n_features:
        sample_row = sample_row[:n_features]
    elif sample_row.shape[0] < n_features:
        sample_row = np.pad(sample_row, (0, n_features - sample_row.shape[0]), 'constant')
    sample = sample_row.reshape(1, -1)
    prediction = best_model.predict(sample)[0]
    
    st.subheader("Tahmin Sonucu")
    color_map = {0: "red", 1: "green", 2: "blue", 3: "orange", 4: "purple"}
    st.markdown(f"<h2 style='color:{color_map.get(prediction, 'black')};'>{prediction + 1}</h2>", unsafe_allow_html=True)
    
    # -------------------------
    # 5️⃣ Interaktif EEG Grafiği (Plotly)
    # -------------------------
    st.subheader("EEG Sinyalleri (Interaktif)")
    fig = go.Figure()
    
    for ch in channels:
        fig.add_trace(go.Scatter(
            y=data_ma[ch],
            name=ch,
            mode='lines'
        ))
    
    # Kriz bölgeleri
    if highlight_crisis:
        # Basit örnek: Tahmin sınıfına göre tüm sinyal 0/1
        # Gerçek kriz segmenti varsa CSV’de label ile değiştirilebilir
        crisis_start, crisis_end = int(len(data)/3), int(len(data)/2)
        fig.add_vrect(x0=crisis_start, x1=crisis_end, fillcolor="red", opacity=0.2,
                      layer="below", line_width=0, annotation_text="Kriz", annotation_position="top left")
    
    fig.update_layout(
        xaxis_title="Zaman",
        yaxis_title="EEG Amplitüdü",
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------
    # 6️⃣ Feature Importance
    # -------------------------
    st.subheader("Feature Importance")
    importances = best_model.feature_importances_
    feature_names = [f"F{i+1}" for i in range(len(importances))]
    
    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=True)
    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        y=feat_imp["feature"],
        x=feat_imp["importance"],
        orientation='h',
        marker_color='teal'
    ))
    fig_imp.update_layout(title="En Önemli 20 Özellik", height=600)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # -------------------------
    # 7️⃣ SHAP Değerleri
    # -------------------------
    st.subheader("SHAP Değerleri (Örnek Satır)")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(sample)
    
    # Plotly desteklenmediği için matplotlib ile göster
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, sample, feature_names=feature_names, plot_type="bar", show=False)
    st.pyplot(plt.gcf())
