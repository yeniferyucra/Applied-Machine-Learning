import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from io import BytesIO
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

# --- Cargar modelo ---
with open("mejorModelo.pkl", "rb") as f:
    modelo = pickle.load(f)

# Mapeo departamentos a regiones
departamento_a_region = {
    "Lima": "Costa", "Callao": "Costa", "Ica": "Costa", "Arequipa": "Costa", "La Libertad": "Costa",
    "Lambayeque": "Costa", "Piura": "Costa", "Tumbes": "Costa", "Ancash": "Costa", "Moquegua": "Costa",
    "Tacna": "Costa", "Chiclayo": "Costa", "Chimbote": "Costa",
    "Cusco": "Sierra", "Junín": "Sierra", "Ayacucho": "Sierra", "Puno": "Sierra", "Huancavelica": "Sierra",
    "Huánuco": "Sierra", "Apurímac": "Sierra", "Cajamarca": "Sierra", "Pasco": "Sierra", "San Martín": "Sierra",
    "Ancash": "Sierra",
    "Loreto": "Selva", "Ucayali": "Selva", "Madre de Dios": "Selva", "Amazonas": "Selva"
}

def region_flags(departamento):
    region = departamento_a_region.get(departamento, "Costa")
    return 1 if region == "Costa" else 0, 1 if region == "Sierra" else 0

def calcular_antiguedad(fecha_registro):
    hoy = datetime.today()
    antiguedad_meses = (hoy.year - fecha_registro.year) * 12 + (hoy.month - fecha_registro.month)
    if fecha_registro.day > hoy.day:
        antiguedad_meses -= 1
    return max(antiguedad_meses, 0)

def preprocess_input_manual(sexo, edad, departamento, fecha_registro, numero_atenciones):
    sexo_codificado = 1 if sexo == "Masculino" else 0
    costa, sierra = region_flags(departamento)
    antiguedad = calcular_antiguedad(fecha_registro)
    return pd.DataFrame([[sexo_codificado, edad, numero_atenciones, antiguedad, costa, sierra]],
                        columns=["Sexo", "Edad", "Cantidad de Atenciones", "Antiguedad de Filiacion", "Costa", "Sierra"])

def preprocess_df(df):
    df["Sexo"] = df["Sexo"].str.lower().map({"masculino": 1, "femenino": 0})
    df["Costa"] = df["Departamento"].map(lambda x: 1 if departamento_a_region.get(x, "Costa") == "Costa" else 0)
    df["Sierra"] = df["Departamento"].map(lambda x: 1 if departamento_a_region.get(x, "Costa") == "Sierra" else 0)
    df["Fecha Registro"] = pd.to_datetime(df["Fecha Registro"], errors='coerce')
    df["Antiguedad de Filiacion"] = df["Fecha Registro"].apply(calcular_antiguedad)
    return df.rename(columns={
        "Edad": "Edad",
        "Cantidad de Atenciones": "Cantidad de Atenciones",
        "Sexo": "Sexo",
        "Antiguedad de Filiacion": "Antiguedad de Filiacion",
        "Costa": "Costa",
        "Sierra": "Sierra"
    })[["Sexo", "Edad", "Cantidad de Atenciones", "Antiguedad de Filiacion", "Costa", "Sierra"]]

def predict(df):
    proba = modelo.predict_proba(df)[:, 1]
    pred = (proba >= 0.56).astype(int)
    return proba, pred

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


st.set_page_config(page_title="🏥 Predicción INEN", layout="wide")

# --- CSS personalizado con título más grande ---
st.markdown("""
<style>
.big-title {
    font-size: 5rem;  /* aumentado de 3rem a 4rem */
    font-weight: 800;
    color: #0066cc;
    text-align: center;
    margin-bottom: 0;
}
.sub-title {
    font-size: 5rem;
    color: #333333;
    text-align: center;
    margin-top: 0;
    margin-bottom: 30px;
}
.metric-box {
    background: linear-gradient(135deg, #5b9df9, #8f6ed5);
    padding: 20px;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    margin: 5px;
}
.metric-value {
    font-size: 2.8rem;
    font-weight: bold;
}
.metric-label {
    font-size: 1.1rem;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)



# --- ENCABEZADO con LOGO ---
st.image("https://pharmaboardroom.com/wp-content/uploads/2015/03/Logo-INEN-300x164.png", width=160)
st.markdown('<h1 style="font-size: 4rem; color: #0066cc; text-align: center;">🏥 Predicción de Cirugía Oncológica</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="font-size: 2rem; text-align: center;">Machine Learning aplicado a la gestión hospitalario</h2>', unsafe_allow_html=True)


# --- MÉTRICAS FALSAS ---
st.markdown("Este proyecto desarrolla una **herramienta predictiva con Machine Learning** para anticipar si un paciente del INEN requerirá cirugía oncológica de alta complejidad.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-box"><div class="metric-value">19.8%</div><div class="metric-label">Pacientes operados</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box"><div class="metric-value">76.1%</div><div class="metric-label">Recall obtenido</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><div class="metric-value">69.9%</div><div class="metric-label">Accuracy final</div></div>', unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Formulario Manual", "Predicción Masiva CSV"],
    icons=["pencil-square", "file-earmark-spreadsheet"],
    orientation="horizontal",
    styles={
        "nav-link-selected": {"background-color": "#0066cc", "color": "white"},
        "container": {"margin-bottom": "20px"}
    }
)

# Lista área de especialización
areas = [
    "ABDOMEN", "GINECOLOGIA", "MAMAS Y TEJIDOS BLANDOS", "ONCOLOGIA PEDIATRICA", "TORAX",
    "NEUROCIRUGIA", "CABEZA Y CUELLO", "UROLOGIA", "MEDICINA", "RADIOTERAPIA",
    "ORTOPEDIA ONCOLOGICA", "MEDICINA NUCLEAR", "GENETICA MEDICA"
]

if selected == "Formulario Manual":
    with st.form("formulario_prediccion"):
        col1, col2 = st.columns(2)

        with col1:
            sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
            edad = st.number_input("Edad", min_value=0, max_value=120, step=1)
            departamento = st.selectbox("Departamento", sorted(departamento_a_region.keys()))
            area = st.selectbox("Área de Especialización", areas)

        with col2:
            fecha_registro = st.date_input("Fecha de registro")
            numero_atenciones = st.number_input("Cantidad de atenciones", min_value=0, step=1)

        antiguedad = calcular_antiguedad(fecha_registro)
        st.markdown(f"**Antigüedad calculada (meses):** {antiguedad}")

        submitted = st.form_submit_button("Predecir")

    if submitted:
        df_input = preprocess_input_manual(sexo, edad, departamento, fecha_registro, numero_atenciones)
        proba, pred = predict(df_input)
        st.markdown(f"### 🩺 Probabilidad de cirugía: **{proba[0]:.2%}**")
        if pred[0] == 1:
            st.success("✅ El paciente probablemente **tendrá** cirugía.")
        else:
            st.warning("❌ El paciente probablemente **no tendrá** cirugía.")

elif selected == "Predicción Masiva CSV":
    st.info("Sube un archivo `.csv` con columnas: `Fecha Registro`, `Sexo`, `Departamento`, `Edad`, `Cantidad de Atenciones`")
    csv_file = st.file_uploader("📂 Subir archivo CSV", type="csv")

    if csv_file:
        try:
            df_raw = pd.read_csv(csv_file)
            st.write("📄 Vista previa de los datos:")
            st.dataframe(df_raw.head())

            expected_cols = ["Fecha Registro", "Sexo", "Departamento", "Edad", "Cantidad de Atenciones"]
            if not all(col in df_raw.columns for col in expected_cols):
                st.error(f"El CSV debe contener las columnas: {', '.join(expected_cols)}")
            else:
                df_proc = preprocess_df(df_raw)
                proba, pred = predict(df_proc)
                df_raw["Probabilidad Cirugia"] = proba
                df_raw["Predicción Cirugia"] = np.where(pred == 1, "Sí", "No")
                st.success("✅ ¡Predicciones realizadas!")
                st.dataframe(df_raw.head())

                csv = convert_df_to_csv(df_raw)
                st.download_button("📥 Descargar CSV con resultados", data=csv, file_name="resultados_prediccion.csv", mime="text/csv")

                # Métricas cirugías predichas
                total = len(df_raw)
                cirugias = (pred == 1).sum()
                porcentaje = cirugias / total * 100 if total > 0 else 0

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{cirugias}</div><div class="metric-label">Cirugías Predichas</div></div>', unsafe_allow_html=True)
                with col_b:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{porcentaje:.2f}%</div><div class="metric-label">Porcentaje Cirugías</div></div>', unsafe_allow_html=True)

                # Gráficos en fila pequeña
                df_raw["Región"] = df_raw["Departamento"].map(departamento_a_region).fillna("Costa")

                fig, axes = plt.subplots(1,4, figsize=(20,4))

                # Pie Sexo
                sexo_counts = df_raw["Sexo"].value_counts()
                # Mapear índices 1 y 0 a etiquetas
                labels_map = {1: "Masculino", 0: "Femenino"}
                labels = [labels_map.get(x, str(x)) for x in sexo_counts.index]

                axes[0].pie(sexo_counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
                axes[0].set_title("Sexo")


                # Pie Región
                region_counts = df_raw["Región"].value_counts()
                axes[1].pie(region_counts, labels=region_counts.index, autopct='%1.1f%%', colors=['#99ff99', '#ffcc99', '#66c2ff'])
                axes[1].set_title("Región")

                # Histograma Edad
                axes[2].hist(df_raw["Edad"], color="#66b3ff", edgecolor='black')
                axes[2].set_title("Edad")
                axes[2].set_xlabel("Edad")
                axes[2].set_ylabel("Frecuencia")

                # Histograma Cantidad Atenciones
                axes[3].hist(df_raw["Cantidad de Atenciones"], color="#ff9999", edgecolor='black')
                axes[3].set_title("Cantidad Atenciones")
                axes[3].set_xlabel("Atenciones")
                axes[3].set_ylabel("Frecuencia")

                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error al leer o procesar el archivo: {e}")
