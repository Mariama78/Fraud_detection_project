import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io

st.set_page_config(page_title="Détection de Fraude en Assurance", layout="wide")
st.title("🚗🔍 Détection de Fraude à l'Assurance Automobile")
st.subheader("Auteur: Mariama Ciré Camara")
st.markdown("""
<style>
    .main {
        background-color: #f9f3d6;
    }
    h1 {
        color: #1E90FF;
    }
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["🧾 Prédiction Tabulaire", "🖼️ Prédiction Image"])

with tabs[0]:
    st.header("🧾 Formulaire de Réclamation")

    with st.form(key='tabular_form'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fault = st.selectbox("Responsabilité (Fault)", ["Policy Holder", "Third Party"])
            base_policy = st.selectbox("Type de Police (BasePolicy)", ["Liability", "Collision", "All Perils"])
            address_change = st.selectbox("Changement d'adresse", ["under 6 months", "1 year", "2 to 3 years", "4 to 8 years", "no change"])
            month = st.selectbox("Mois de l'accident", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        
        with col2:
            days_policy_claim = st.selectbox("Jours: Police-Réclamation", ["15 to 30", "8 to 15", "more than 30", "none"])
            month_claimed = st.selectbox("Mois de réclamation", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            year = st.selectbox("Année", [1994, 1995, 1996])
            number_of_cars = st.selectbox("Nombre de véhicules", ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"])
        
        with col3:
            make = st.selectbox("Marque du véhicule (Make)", ["Honda", "Toyota", "Ford", "BMW", "Mazda", "VW", "Ferrari", "Jaguar", "Chevrolet", "Dodge"])
            police_report = st.selectbox("Rapport de police disponible?", ["Yes", "No"])

        submitted = st.form_submit_button("🔍 Prédire")

        if submitted:
            input_data = {
                "Fault": fault,
                "BasePolicy": base_policy,
                "AddressChange-Claim": address_change,
                "Month": month,
                "Days:Policy-Claim": days_policy_claim,
                "MonthClaimed": month_claimed,
                "Year": year,
                "NumberOfCars": number_of_cars,
                "Make": make,
                "PoliceReportFiled": police_report
            }

            response = requests.post("http://127.0.0.1:5000/predict/tabulaire", json=input_data)
            if response.status_code == 200:
                st.success(f"Résultat : {response.json()['prediction']}")
            else:
                st.error("Erreur côté backend : " + response.text)


with tabs[1]:
    st.header("🖼️ Analyse d'une Image de Réclamation")
    uploaded_image = st.file_uploader("Choisissez une image d'accident (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Image chargée", use_column_width=True)
        if st.button("🔍 Prédire depuis Image"):
            files = {"image": uploaded_image.getvalue()}
            response = requests.post("http://127.0.0.1:5000/predict/image", files={"image": uploaded_image})

            if response.status_code == 200:
                result = response.json()
                st.success(f"Résultat : {result['prediction']} (Probabilité : {result['probabilité_fraude']:.2f})")
            else:
                st.error("Erreur côté backend : " + response.text)
