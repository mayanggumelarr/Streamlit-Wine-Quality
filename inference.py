import joblib
import numpy as np
import pandas as pd
import streamlit as st

scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("model.pkl")

# Image of Wine
st.image("wine.jpeg", caption="Wine and Sunset")

# Title
st.title("Klasifikasi Qualitas Wine")
st.write("Masukkan hasil pengamatan/pengukuran untuk klasifikasi!")
st.write("Catatan: Masukkan dalam format desimal")

# Inputan
alcohol = st.number_input("Alcohol (0.0 - 15.0): ", min_value=0.0, max_value=15.0)
malic = st.number_input("Malic Acid (0.0 - 10.0): ", min_value=0.0, max_value=10.0)
ash = st.number_input("Ash (0.0 - 5.0): ", min_value=0.0, max_value=5.0)
alcalin_ash = st.number_input("Alcalinity of Ash (0.0 - 30.0): ", min_value=0.0, max_value=30.0)
magnesium = st.number_input("Magnesium (0.0 - 170.0): ", min_value=0.0, max_value=170.0)
phenols = st.number_input("Total Phenols (0.0 - 4.0): ", min_value=0.0, max_value=4.0)
flavanoids = st.number_input("Flavanoids (0.0 - 6.0): ", min_value=0.0, max_value=6.0)
nonflavanoiods = st.number_input("Nonflavanoids Phenols (0.0 - 1.0): ", min_value=0.0, max_value=1.0)
proan = st.number_input("Proanthocyanins (0.0 - 4.0): ", min_value=0.0, max_value=4.0)
color = st.number_input("Color Intensity (0.0 - 14.0): ", min_value=0.0, max_value=14.0)
hue = st.number_input("Input Hue (0.0 - 2.0): ", min_value=0.0, max_value=2.0)
od = st.number_input("Od280/Od315_of_diluted_wines (0.0 - 4.0): ", min_value=0.0, max_value=4.0)
proline = st.number_input("Proline (0.0 - 1700.0): ", min_value=0.0, max_value=1700.0)

if st.button("Predict Wine Quality", type="primary"):

    # 'value (like column in df)' : [var]
    input_data_dict = {
        'alcohol': [alcohol],
        'malic_acid': [malic],
        'ash': [ash],
        'alcalinity_of_ash': [alcalin_ash],
        'magnesium': [magnesium],
        'total_phenols': [phenols],
        'flavanoids': [flavanoids],
        'nonflavanoid_phenols': [nonflavanoiods],
        'proanthocyanins': [proan],
        'color_intensity': [color],
        'hue': [hue],
        'od280/od315_of_diluted_wines': [od],
        'proline': [proline]
    }

    df = pd.DataFrame(input_data_dict)

    # Scale
    df_scaled = scaler.transform(df)

    # PCA
    df_PCA = pca.transform(df_scaled)

    # Predict
    prediction = model.predict(df_PCA)
    result = prediction[0]

    if result == 0:
        st.write("High Quality")
    elif result == 1:
        st.write("Medium Quality")
    else:
        st.write("Low Quality")

st.link_button("Project's Repository", url='https://github.com/mayanggumelarr/Streamlit-Wine-Quality')
