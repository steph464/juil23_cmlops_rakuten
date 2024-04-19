# predict_router.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from PIL import Image
from io import BytesIO
from typing import List
import os
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from fastapi import UploadFile, Depends
from user.user_router import authenticate_user  # Importez votre fonction d'authentification ici

from predict.model import load_modelresnet, predict, prepare_image
import pandas as pd
import pickle
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from predict.texte_process import CreateTextANDcleaning
import cv2
from fastapi.responses import JSONResponse
import base64


predict_router = APIRouter()

# Définissez la classe de réponse JSON
class Prediction(BaseModel):
    filename: str
    content_type: str
    predictions: List[dict] = []

@predict_router.post("/predict_image_class resnet", response_model=Prediction)
async def prediction(
    file: UploadFile = File(...),
    current_user: str = Depends(authenticate_user),
):
    
    # Chargez le modèle de classification (vous devez implémenter cette fonction dans un fichier séparé)
    model = load_modelresnet()
        
    # Assurez-vous que le fichier est une image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier fourni n'est pas une image.")
    
    content = await file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    
    # Prétraitement de l'image et préparation pour la classification
    image = prepare_image(image, target=(224, 224))
    
    
    
    # Effectuez la prédiction en utilisant le modèle
    response = predict(image, model)
 # Vous devez implémenter cette fonction
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "predictions": response,
    }



@predict_router.post("/predict_text_class")

async def predict_text_class(designation: str, 
                             description: str, 
                             current_user: str = Depends(authenticate_user)):
    
    # Créez un DataFrame à partir des données d'entrée
    data = {'designation': [designation], 'description': [description]}
    df = pd.DataFrame(data)

    # Effectuez le prétraitement sur les données textuelles
    processed_text = CreateTextANDcleaning(df)

    # Chargez le tokenizer pré-entraîné
    tokenizer = Tokenizer()
    
    # with open('/home/workspace/API/app/models/fitted_tokenizer.pickle', 'rb') as handle:
    #fitted_tokenizer = pickle.load(handle)
    

    # Appliquez le tokenizer sur le texte
    tokenizer.fit_on_texts(processed_text)
    
    maxlen = 400 
    text = tokenizer.texts_to_sequences(processed_text)
    
    #text = fitted_tokenizer.texts_to_sequences(processed_text)
    text = keras.preprocessing.sequence.pad_sequences(text, maxlen=maxlen, padding='post')

    #Chargez le modèle de classification textuelle
    model_conv1D = keras.models.load_model('/home/workspace/API/app/models/conv1d_model.h5')

    #Effectuez la prédiction
    y_pred_samples = model_conv1D.predict(text)
    
    ## classe prédite en utilisant argmax
    predictedtexte_class = np.argmax(y_pred_samples, axis=1)
    
    # la probabilité de chaque classe
    classtexte_probabilities = y_pred_samples[0]
    
    # Liste des labels
    labels = ["Livres adultes", "Jeux Vidéos", "Accessoires de Jeux Vidéos", "Consoles de jeux", "Figurine", "Carte à Collectionner",
            "Masques", "Jouets pour Enfants", "Jeux de Cartes et de société", "Produits télécommandés",
            "Vêtements pour enfants", "Jouets pour Enfants", "Produits Bébés et Enfants",
            "Literies et Meubles", "Accessoires Maison", "Alimentation", "Décoration d'intérieur", "Accessoires Animaux",
            "Journaux et Magazines", "Livres et Revues", "Jeux", "Papeterie",
            "Mobilier", "Piscine", "Jardinage", "Livres", "Jeux en ligne et Logiciels"]
    
    # Code produit associé aux classes
    code_produit = [10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281,
                1300, 1301, 1302, 1320, 1560, 1920, 1940, 2060,
                2220, 2280, 2403, 2462, 2522, 2582, 2583,
                2585, 2705, 2905]
    
    

    # Association des classes prédites aux labels et au code produit
    predicted_label = labels[predictedtexte_class[0]]
    predicted_code = code_produit[predictedtexte_class[0]]

    # Créez une réponse JSON pour afficher les résultats
    response_data = {
        "predicted_class": predicted_label,
        "class_probabilities": classtexte_probabilities.tolist(),
        "predicted_code": predicted_code
    }

    return JSONResponse(content=response_data)
    

@predict_router.post("/predict_image_class")
async def predict_image_class(
    file: UploadFile = File(...),
    current_user: str = Depends(authenticate_user),
):
            
    # chargement du modèle pré-entraîné xception
    model = load_model('/home/workspace/API/app/models/checkpoint_Xception_model.h5')
        
    # Assurez-vous que le fichier est une image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier fourni n'est pas une image.")
    
    
    # Lisez le contenu du fichier
    content = await file.read()
    
    # Lisez l'image à l'aide de OpenCV
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    
    # Assurez-vous que l'image a les dimensions requises (299x299)
    image = cv2.resize(image, (299, 299))
    
    # Normalisation de l'image
    image = image / 255.0
    
    # Ajout d'une dimension pour correspondre aux attentes du modèle
    image = np.expand_dims(image, axis=0)
    
    # prédiction avec le modèle xception
    predictions = model.predict(image)

    ## classe prédite en utilisant argmax
    predicted_class = np.argmax(predictions, axis=1)

    # la probabilité de chaque classe
    class_probabilities = predictions[0]
    
    
    # Liste des labels
    labels = ["Livres adultes", "Jeux Vidéos", "Accessoires de Jeux Vidéos", "Consoles de jeux", "Figurine", "Carte à Collectionner",
            "Masques", "Jouets pour Enfants", "Jeux de Cartes et de société", "Produits télécommandés",
            "Vêtements pour enfants", "Jouets pour Enfants", "Produits Bébés et Enfants",
            "Literies et Meubles", "Accessoires Maison", "Alimentation", "Décoration d'intérieur", "Accessoires Animaux",
            "Journaux et Magazines", "Livres et Revues", "Jeux", "Papeterie",
            "Mobilier", "Piscine", "Jardinage", "Livres", "Jeux en ligne et Logiciels"]
    
    # Code produit associé aux classes
    code_produit = [10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281,
                1300, 1301, 1302, 1320, 1560, 1920, 1940, 2060,
                2220, 2280, 2403, 2462, 2522, 2582, 2583,
                2585, 2705, 2905]
    
    
    # Association des classes prédites aux labels et au code produit
    predicted_label = labels[predicted_class[0]]
    predicted_code = code_produit[predicted_class[0]]

    # Créez une réponse JSON pour afficher les résultats
    response_data = {
        "predicted_class": predicted_label,
        "class_probabilities": class_probabilities.tolist(),
        "predicted_code": predicted_code
    }

    return JSONResponse(content=response_data)

@predict_router.post("/predict_bimodal_class")

async def predict_bimodal_class(designation: str, 
                             description: str,
                             file: UploadFile = File(...),
                             current_user: str = Depends(authenticate_user)):
    
    # Assurez-vous que le fichier est une image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier fourni n'est pas une image.")
    
    
    # Lisez le contenu du fichier image
    content = await file.read()
    
    # Créez un DataFrame à partir des données d'entrée description et designation
    data = {'designation': [designation], 'description': [description]}
    df = pd.DataFrame(data)
    
    ################################### traitement du texte #######################################################
    
    # Effectuez le prétraitement sur les données textuelles
    processed_text = CreateTextANDcleaning(df)

    # Chargez le tokenizer pré-entraîné
    tokenizer = Tokenizer()
    
    # Appliquez le tokenizer sur le texte
    tokenizer.fit_on_texts(processed_text)
    
    maxlen = 400 
    text = tokenizer.texts_to_sequences(processed_text)
    
    text = keras.preprocessing.sequence.pad_sequences(text, maxlen=maxlen, padding='post')

    #Chargez le modèle de classification textuelle
    model_conv1D = keras.models.load_model('/home/workspace/API/app/models/conv1d_model.h5')

    #Effectuez la prédiction
    y_pred_samples = model_conv1D.predict(text)
    ################################################traitement de l'image########################################
    
    # Lisez l'image à l'aide de OpenCV
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    
    # Assurez-vous que l'image a les dimensions requises (299x299)
    image = cv2.resize(image, (299, 299))
    
    # Normalisation de l'image
    image = image / 255.0
    
    # Ajout d'une dimension pour correspondre aux attentes du modèle
    image = np.expand_dims(image, axis=0)
    
    # chargement du modèle pré-entraîné xception
    model = load_model('/home/workspace/API/app/models/checkpoint_Xception_model.h5')
    
    # prédiction avec le modèle xception
    predictionsimage = model.predict(image)

    ## classe prédite en utilisant argmax
    predicted_class = np.argmax(predictionsimage, axis=1)

    # la probabilité de chaque classe
    class_probabilities = predictionsimage[0]
    
    
    
    ##########################################"résultat" ########################################################"
    
    
    ## classe texte prédite en utilisant argmax
    predictedtexte_class = np.argmax(y_pred_samples, axis=1)
    
    # la probabilité de chaque classe de texte
    classtexte_probabilities = y_pred_samples[0]
    
    ## classe image prédite en utilisant argmax
    predictedimage_class = np.argmax(predictionsimage, axis=1)

    # la probabilité de chaque classe
    classimage_probabilities = predictionsimage[0]
    
    
    # Liste des labels
    labels = ["Livres adultes", "Jeux Vidéos", "Accessoires de Jeux Vidéos", "Consoles de jeux", "Figurine", "Carte à Collectionner",
            "Masques", "Jouets pour Enfants", "Jeux de Cartes et de société", "Produits télécommandés",
            "Vêtements pour enfants", "Jouets pour Enfants", "Produits Bébés et Enfants",
            "Literies et Meubles", "Accessoires Maison", "Alimentation", "Décoration d'intérieur", "Accessoires Animaux",
            "Journaux et Magazines", "Livres et Revues", "Jeux", "Papeterie",
            "Mobilier", "Piscine", "Jardinage", "Livres", "Jeux en ligne et Logiciels"]
    
    # Code produit associé aux classes
    code_produit = [10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281,
                1300, 1301, 1302, 1320, 1560, 1920, 1940, 2060,
                2220, 2280, 2403, 2462, 2522, 2582, 2583,
                2585, 2705, 2905]
    
    
    # Association des classes prédites aux labels et au code produit
    predictedtexte_label = labels[predictedtexte_class[0]]
    predictedtexte_code = code_produit[predictedtexte_class[0]]
    
    # Association des classes prédites aux labels et au code produit
    predictedimage_label = labels[predictedimage_class[0]]
    predictedimage_code = code_produit[predictedimage_class[0]]
    
    
    # Créez une réponse JSON pour afficher les résultats
    response_data = {
        "texte predicted_class": predictedtexte_label,
        "classtexte_probabilities": classtexte_probabilities.tolist(),
        "predictedtexte_code": predictedtexte_code,
        "predictedimage_class": predictedimage_label,
        "classimage_probabilities": classimage_probabilities.tolist(),
        "predictedimage_code": predictedimage_code
    }
    
    return JSONResponse(content=response_data)
    

    
    