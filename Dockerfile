# image de base
FROM python:3.11-slim

# créer un répertoire de travail
WORKDIR /app

# Copier les fichiers du projet dans l’image
COPY . .

# Installer les dépendances
RUN pip install --upgrade pip && pip install -r requirements.txt

# Exposer le port sur lequel Flask sera lancé
EXPOSE 5000

# Lancer l'application Flask
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]