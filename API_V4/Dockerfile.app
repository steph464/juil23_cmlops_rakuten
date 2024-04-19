# Image consruite le 11/11/2023 modifiée le 13/11/2023

# Utilisez une image de base Python 3.9.18
#FROM python:3.9.18
FROM python:3.9

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez les fichiers de l'application dans le conteneur
#COPY ./app_V4/main_router.py /app/
#COPY ./app_V4/. /app/
COPY . .

# Ajout des instructions pour installer les dépendances système   le 13/11/2023 api_rakuten1 docker images
#RUN apt-get update && apt-get install -y \
#    libgl1-mesa-glx \
#    # Autres dépendances système éventuelles
#    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    apt-get install -y \
    libgl1-mesa-glx


#RUN apt-get update && apt-get install -y

# Installez les dépendances
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Exposez le port sur lequel l'application s'exécute 
EXPOSE 8000

# commande pour exécuter l'application
#CMD ["python", "./main_router.py"]
CMD ["uvicorn", "main_router:api", "--host", "0.0.0.0", "--port", "8000"]