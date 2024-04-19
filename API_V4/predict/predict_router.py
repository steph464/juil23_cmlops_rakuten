# predict_router.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from PIL import Image
from io import BytesIO
from typing import List
import os
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import UploadFile, Depends
from user.user_router import authenticate_user  # Importez votre fonction d'authentification ici
import datetime
from predict.model import load_modelresnet, predict, prepare_image
import pandas as pd
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from predict.texte_process import CreateTextANDcleaning
import cv2
from fastapi.responses import JSONResponse
import uuid


predict_router = APIRouter()

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


########################################################################################################""

@predict_router.post("/predict_text_class")

async def predict_text_class(designation: str, 
                             description: str, 
                             current_user: str = Depends(authenticate_user)):
    
    # Créez un DataFrame à partir des données d'entrée
    data = {'designation': [designation], 'description': [description]}
    df = pd.DataFrame(data)

    # Effectuez le prétraitement sur les données textuelles
    text_to_predict= CreateTextANDcleaning(df)
    #text_to_predict=text_to_df[0]
    

    # Tokenisation du texte
    tokenizer = Tokenizer(num_words=20000)  # Assurez-vous que le tokenizer est le même que celui utilisé lors de l'entraînement
    tokenizer.fit_on_texts(text_to_predict)
    text_sequences = tokenizer.texts_to_sequences(text_to_predict)

    # Padding des séquences
    maxlen = 400  # Assurez-vous que c'est la même longueur maximale que celle utilisée lors de l'entraînement
    text_sequences = pad_sequences(text_sequences, padding='post', maxlen=maxlen)
    
    #Chargez le modèle de classification textuelle
    model_conv1D = keras.models.load_model(valid_text_model_path)
    #Effectuez la prédiction
    y_pred_samples = model_conv1D.predict(text_sequences)
    y_pred_texte= y_pred_samples * 100
    
    ## classe prédite en utilisant argmax
    predictedtexte_class = np.argmax(y_pred_texte, axis=1)
    
    # la probabilité de chaque classe
    classtexte_probabilities = y_pred_texte[0]
    
    
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

####################################################################################################
    # Création d'un DataFrame pour stocker les données
    data_to_save = {
            'processed_text': text_to_predict,
            'text_predictions': y_pred_texte.tolist(),
            "predicted_code": predicted_code,
            }
    df_outpout = pd.DataFrame(data_to_save)
    # Obtenez la date et l'heure actuelles
    current_datetime = datetime.datetime.now()
    # Obtenez un numéro unique
    unique_id = str(uuid.uuid4())
    
    # Ajoutez la date et l'heure à df_outpout
    df_outpout['enregistrement_date'] = current_datetime.date()
    df_outpout['enregistrement_heure'] = current_datetime.time()
    df_outpout['utilisateur'] = current_user
    df_outpout['unique_id'] = unique_id
        
    # Sauvegarde des données dans un fichier CSV
    #df_outpout.to_csv('/home/workspace/API/BDD/data_outpout.csv', index=False)
    
    # Charger le DataFrame existant (Texte_data)
    Texte_data = pd.read_csv(text_data_path)

    # Concaténer le DataFrame df_outpout avec Texte_data
    Texte_data = pd.concat([Texte_data, df_outpout], ignore_index=True)

    # Sauvegarder le DataFrame Texte_data dans un fichier CSV
    Texte_data.to_csv(text_data_path, index=False)
    
    # Créez une réponse JSON pour afficher les résultats
    response_data = {
        "unique_id":unique_id,
        "predicted_class": predicted_label,
        "predicted_code": predicted_code,
        "class_probabilities": classtexte_probabilities.tolist(),
        }
    
    return JSONResponse(content=response_data)
    

########################################################################################################
########################################################################################################

@predict_router.post("/predict_image_class")
async def predict_image_class(
    file: UploadFile = File(...),
    current_user: str = Depends(authenticate_user),
):
            
    # chargement du modèle pré-entraîné xception
    model = load_model(valid_img_model_path)
        
    # Assurez-vous que le fichier est une image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier fourni n'est pas une image.")
    
    
    # Lisez le contenu du fichier
    content = await file.read()
    
    #####################################################################################################
    
    # Lisez l'image à l'aide de OpenCV
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    
    # Assurez-vous que l'image a les dimensions requises (299x299)
    image = cv2.resize(image, (299, 299))
    
    # Normalisation de l'image
    image = image / 255.0
    
    # Ajout d'une dimension pour correspondre aux attentes du modèle
    image = np.expand_dims(image, axis=0)
    
    # prédiction avec le modèle xception
    predictionsimage = model.predict(image)
    y_pred_image= predictionsimage * 100
    
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
    
    ## classe prédite en utilisant argmax
    predictedimg_class = np.argmax(y_pred_image, axis=1)
    
    # Association des classes prédites aux labels et au code produit
    predicted_label = labels[predictedimg_class[0]]
    predicted_code = code_produit[predictedimg_class[0]]
    
    # fonction reshape pour convertir en tableau 1D
    prob_class_img = (y_pred_image.reshape(-1))
    
    # Création du DataFrame
    data = {'code_produit': code_produit, 'labels': labels, 'prob_class_img': prob_class_img}
    df = pd.DataFrame(data)
    
    ##################################################################################################
    
        
    # Obtenez la date et l'heure actuelles et un numéro unique
    current_datetime = datetime.datetime.now()
    unique_id = str(uuid.uuid4())
    

    # Indice de la valeur maximale pour weighted_proba
    idx_max = df['prob_class_img'].idxmax()
    max_row = df.loc[idx_max].to_dict()
        
    # Ajoutez le numéro unique à max_weighted_row
    max_row['unique_id'] = unique_id
    
    ####################################################################################################
    ###################### récupération de l'image  ################################################
    # Récupérez le nom du fichier d'origine
    original_filename = file.filename

    # Générez le chemin complet de sauvegarde avec le nom du fichier
    image_path = os.path.join(img_data_path, original_filename)

    # Sauvegardez l'image avec le nom du fichier d'origine
    cv2.imwrite(image_path, image[0] * 255)  # Rétablissez la plage de valeurs [0, 255]
    
    # Création d'un DataFrame pour stocker les données
    data_to_save = {
            'predicted code':predicted_code,
            'predicted label':predicted_label,
            'image_predictions': y_pred_image.tolist()
            }
    
    df_img = pd.DataFrame(data_to_save)
    
    # Ajoutez la date et l'heure à df_outpout
    df_img['enregistrement_date'] = current_datetime.date()
    df_img['enregistrement_heure'] = current_datetime.time()
    df_img['utilisateur'] = current_user
    # Ajoutez le chemin de l'image au DataFrame df_bimod
    df_img['image_path'] = image_path
    df_img['unique_id']= unique_id
    
    # Sauvegarde des données dans un fichier CSV
    #img_data.to_csv('/home/workspace/API/API_rakuten/BDD/dataset/valid_data/img_data.csv', index=False)
    
    # Charger le DataFrame existant (bimod_data)
    img_data = pd.read_csv(img_dataset_path)

    #Concaténer le DataFrame df_outpout avec bimod_data
    img_data_data = pd.concat([df_img, img_data], ignore_index=True)

    # Sauvegarder le DataFrame Texte_data dans un fichier CSV
    img_data_data.to_csv(img_dataset_path, index=False)
    
    
    # Créez une réponse JSON pour afficher les résultats
    response_data = {
        "unique_id":unique_id,
        "predicted_class": predicted_label,
        "predicted_code": predicted_code,
        "class_probabilities": y_pred_image.tolist(),
        }
    
    return JSONResponse(content=response_data)


######################################################################################################
######################################################################################################

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
    model_conv1D = keras.models.load_model(valid_text_model_path)

    #Effectuez la prédiction
    y_pred_samples = model_conv1D.predict(text)
    y_pred_texte= y_pred_samples * 100
    
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
    model = load_model( valid_img_model_path)
    
    # prédiction avec le modèle xception
    predictionsimage = model.predict(image)
    y_pred_image= predictionsimage * 100
    

    ## classe prédite en utilisant argmax
    predicted_class = np.argmax(y_pred_image, axis=1)

    # la probabilité de chaque classe
    class_probabilities = y_pred_image[0]
    
    
    
    ##########################################"résultat" ########################################################"
    
    ## classe texte prédite en utilisant argmax
    predictedtexte_class = np.argmax(y_pred_texte, axis=1)
    
    # la probabilité de chaque classe de texte
    classtexte_probabilities = y_pred_texte[0]
    
    ## classe image prédite en utilisant argmax
    predictedimage_class = np.argmax(y_pred_image, axis=1)

    # la probabilité de chaque classe
    classimage_probabilities = y_pred_image[0]
    
    
    #Poids pour le modèle Conv1D
    conv1D_weight = 0.6
    # Poids pour le modèle Xception
    xception_weight = 0.4
    
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
    
    
    # fonction reshape pour convertir en tableau 1D
    prob_class_text = (y_pred_texte.reshape(-1))
    prob_class_img = (y_pred_image.reshape(-1))
    
    

    # Création du DataFrame
    data = {'code_produit': code_produit, 'labels': labels, 'prob_class_text': prob_class_text, 'prob_class_img': prob_class_img}
    df = pd.DataFrame(data)
    ####################################################################################################""
    
    
    df['Pb_pond_text']=df['prob_class_text'] * conv1D_weight 
    df['Pb_Pond_im']=df['prob_class_img']*xception_weight
    df['weighted_proba'] = (df['prob_class_text'] * conv1D_weight + df['prob_class_img'] * xception_weight) /(conv1D_weight + xception_weight)
    
    # Obtenez la date et l'heure actuelles et un numéro unique
    current_datetime = datetime.datetime.now()
    unique_id = str(uuid.uuid4())
    

    # Indice de la valeur maximale pour weighted_proba
    idx_max_wgt = df['weighted_proba'].idxmax()
    max_weighted_row = df.loc[idx_max_wgt].to_dict()
        
    # Ajoutez le numéro unique à max_weighted_row
    max_weighted_row['unique_id'] = unique_id
    
    
    ####################################################################################################
    ###################### récupération de l'image  ################################################
    # Récupérez le nom du fichier d'origine
    original_filename = file.filename

    # Générez le chemin complet de sauvegarde avec le nom du fichier
    image_path = os.path.join(bimod_img_path, original_filename)

    # Sauvegardez l'image avec le nom du fichier d'origine
    cv2.imwrite(image_path, image[0] * 255)  # Rétablissez la plage de valeurs [0, 255]
    
    
    ###################################################################################################
    # Création d'un DataFrame pour stocker les données
    data_to_save = {
            'processed_text': processed_text,
            'text_predictions': y_pred_texte.tolist(),
            'image_predictions': y_pred_image.tolist()
            }
    
    df_bimod = pd.DataFrame(data_to_save)
    
    # Ajoutez la date et l'heure à df_outpout
    df_bimod['enregistrement_date'] = current_datetime.date()
    df_bimod['enregistrement_heure'] = current_datetime.time()
    df_bimod['utilisateur'] = current_user
    # Ajoutez le chemin de l'image au DataFrame df_bimod
    df_bimod['image_path'] = image_path
    df_bimod['unique_id']= unique_id

    # Sauvegarde des données dans un fichier CSV
    #df_bimod.to_csv('/home/workspace/API/BDD/df_bimod.csv', index=False)
    
    # Charger le DataFrame existant (bimod_data)
    bimod_data = pd.read_csv(bimod_data_path)

    #Concaténer le DataFrame df_outpout avec bimod_data
    bimod_data = pd.concat([bimod_data, df_bimod], ignore_index=True)

    # Sauvegarder le DataFrame Texte_data dans un fichier CSV
    bimod_data.to_csv(bimod_data_path, index=False)
    
    ##################################################################################################
    return JSONResponse(content= max_weighted_row)