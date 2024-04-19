# predict_router.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
import os
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import UploadFile, Depends
from user.user_router import authenticate_user  # Importez votre fonction d'authentification ici
import datetime
import pandas as pd
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from predict.texte_process import CreateTextANDcleaning
import cv2
from fastapi.responses import JSONResponse
import uuid


operations = APIRouter()

# Définissez la classe de réponse JSON
class Prediction(BaseModel):
    filename: str
    content_type: str
    predictions: List[dict] = []


###################################" chemins des modèles et dataframe ###################################
    
# chemin du modèle texte valide    
#valid_text_model_path = "/home/workspace/API/API_rakuten/models/text_models/text_valid_models/conv1D_valid_model.h5"
valid_text_model_path = "../models/text_models/text_valid_models/conv1D_valid_model.h5"

# chemin du dataset de sauvegarde Texte.data valid
#text_data_path = "/home/workspace/API/API_rakuten/BDD/dataset/valid_data/Texte_data.csv"
text_data_path = "../BDD/dataset/valid_data/Texte_data.csv"

#chemin du modèle image valide
#valid_img_model_path="/home/workspace/API/API_rakuten/models/img_models/img_valid_models/Xception_valid_model.h5"
valid_img_model_path="../models/img_models/img_valid_models/Xception_valid_model.h5"

# chemin de sauvegarde des images prédites
#img_data_path="/home/workspace/API/API_rakuten/BDD/saved_img_data/Xception_img"
img_data_path="../BDD/saved_img_data/Xception_img"

# chemin du dataset de sauvegarde de img_data valid
#img_dataset_path="/home/workspace/API/API_rakuten/BDD/dataset/valid_data/img_data.csv"
img_dataset_path="../BDD/dataset/valid_data/img_data.csv"

# chemin de sauvegarde de bimod_data, dataset du traitemetn bimodal
#bimod_data_path= "/home/workspace/API/API_rakuten/BDD/dataset/valid_data/bimod_data.csv"
bimod_data_path= "../BDD/dataset/valid_data/bimod_data.csv"

# chemin de sauvegarde des images du traitement bimodal
#bimod_img_path="/home/workspace/API/API_rakuten/BDD/saved_img_data/bimod_img"
bimod_img_path="../BDD/saved_img_data/bimod_img"

########################################################################################################

# Endpoint pour récupérer des données en fonction du numéro unique pour le traitement texte
@operations.get("/get_texte_data_by_id/{unique_id}")
async def get_text_data_by_id(unique_id: str, current_user: str = Depends(authenticate_user)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionalité.")
    
    # Charger le DataFrame existant (Texte_data)
    Texte_data = pd.read_csv(text_data_path)
    
    # Recherchez la ligne correspondant au numéro unique donné
    data_row = Texte_data[Texte_data['unique_id'] == unique_id]
    
    if data_row.empty:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
    # Remplacer les NaN par des valeurs par défaut (par exemple, une chaîne vide)
    data_row = data_row.fillna('')
    
    # Convertir la ligne en un dictionnaire JSON
    data_dict = data_row.iloc[0].to_dict()
    
    return data_dict



####################################################################################################

@operations.put("/update_texte_prediction_status")
async def update_texte_prediction_status(unique_id: str, is_correct: bool, current_user: str = Depends(authenticate_user)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionalité.")
    
    # Charger le DataFrame existant (bimod_data)
    Texte_data = pd.read_csv(text_data_path)
    
    # Recherchez la ligne correspondant au numéro unique donné
    data_row_index = Texte_data[Texte_data['unique_id'] == unique_id].index
    
    if len(data_row_index) == 0:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
    # Mettre à jour le statut de prédiction
    Texte_data.at[data_row_index[0], 'is_correct'] = is_correct
    
    # Sauvegarder le DataFrame mis à jour
    Texte_data.to_csv(text_data_path, index=False)
    
    return {"message": f"Statut de prédiction mis à jour pour l'ID unique {unique_id} : is_correct={is_correct}"}


#########################################################################################################

# Endpoint pour récupérer des données en fonction du numéro unique pour le traitement de l'image
@operations.get("/get_image_data_by_id/{unique_id}")
async def get_img_data_by_id(unique_id: str, current_user: str = Depends(authenticate_user)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionalité.")
    
    # Charger le DataFrame existant (img_data)
    img_data = pd.read_csv(img_dataset_path)
    
    # Recherchez la ligne correspondant au numéro unique donné
    data_row = img_data[img_data['unique_id'] == unique_id]
    
    if data_row.empty:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
    # Remplacer les NaN par des valeurs par défaut (par exemple, une chaîne vide)
    data_row = data_row.fillna('')
    
    # Convertir la ligne en un dictionnaire JSON
    data_dict = data_row.iloc[0].to_dict()
    
    return data_dict


#########################################################################################################


@operations.put("/update_image_prediction_status")
async def update_image_prediction_status(unique_id: str, is_correct: bool, current_user: str = Depends(authenticate_user)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionalité.")
    
    # Charger le DataFrame existant (img_data)
    img_data = pd.read_csv(img_dataset_path)
    
    # Recherchez la ligne correspondant au numéro unique donné
    data_row_index = img_data[img_data['unique_id'] == unique_id].index
    
    if len(data_row_index) == 0:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
    # Mettre à jour le statut de prédiction
    img_data.at[data_row_index[0], 'is_correct'] = is_correct
    
    # Sauvegarder le DataFrame mis à jour
    img_data.to_csv(img_dataset_path, index=False)
    
    return {"message": f"Statut de prédiction mis à jour pour l'ID unique {unique_id} : is_correct={is_correct}"}

#########################################################################################################



# Endpoint pour récupérer des données en fonction du numéro unique
@operations.get("/get_bimod_data_by_id/{unique_id}")
async def get_bimod_data_by_id(unique_id: str, current_user: str = Depends(authenticate_user)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionalité.")
    
    # Charger le DataFrame existant (bimod_data)
    bimod_data = pd.read_csv(bimod_data_path)
    
    # Recherchez la ligne correspondant au numéro unique donné
    data_row = bimod_data[bimod_data['unique_id'] == unique_id]
    
    if data_row.empty:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
        
    # Remplacer les NaN par des valeurs par défaut (par exemple, une chaîne vide)
    data_row = data_row.fillna('')
    
    # Convertir la ligne en un dictionnaire JSON
    data_dict = data_row.iloc[0].to_dict()
    
    return data_dict

########################################################################################################

@operations.put("/update_bimodal_prediction_status")
async def update_bimodal_prediction_status(unique_id: str, is_correct: bool, current_user: str = Depends(authenticate_user)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionalité.")
    
    # Charger le DataFrame existant (bimod_data)
    bimod_data = pd.read_csv(bimod_data_path)
    
    # Recherchez la ligne correspondant au numéro unique donné
    data_row_index = bimod_data[bimod_data['unique_id'] == unique_id].index
    
    if len(data_row_index) == 0:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
    # Mettre à jour le statut de prédiction
    bimod_data.at[data_row_index[0], 'is_correct'] = is_correct
    
    # Sauvegarder le DataFrame mis à jour
    bimod_data.to_csv(bimod_data_path, index=False)
    
    return {"message": f"Statut de prédiction mis à jour pour l'ID unique {unique_id} : is_correct={is_correct}"}
