# image de base
FROM python:3.11-slim

# créer un répertoire de travail
WORKDIR /app

#copier tous les fichiers dans le conteneur
COPY . .

# installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# exposer les ports pour Flask et Streamlit
EXPOSE 5000
EXPOSE 8501

# commande par défaut 
CMD ["streamlit", "run", "streamlit_app.py"]
