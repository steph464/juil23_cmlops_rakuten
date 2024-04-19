import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import PIL
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, classification_report
from keras.models import load_model, Model

# Définition des répertoires et des chemins
images_dir_train = '/home/workspace/API/API_rakuten/BDD/RAKUTEN/images/image_train'

# chemin du modèle Xception valide
f1_score_valid_model_path = '/home/workspace/API/API_rakuten/BDD/modeles/image/img_valid_model/img_valid_model.h5'


def f1_img_test(dataframe):
    
    # Chargement des données
    df = dataframe

    # Conversion de la colonne 'prdtypecode' en string
    df['prdtypecode'] = df['prdtypecode'].astype(str)
    
    # Séparation des données en ensembles d'entraînement et de validation
    X_train, X_test = train_test_split(df, train_size=0.99, random_state=1234)

    # Paramètres du modèle
    img_rows = 299
    img_cols = 299
    batch_size = 64
    dropout_rate = 0.4
    l1 = 0.001

    # Générateurs de données
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        rotation_range=45,
        zoom_range=0.2,
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=X_train,
        directory=images_dir_train,
        x_col='image_name',
        y_col='prdtypecode',
        class_mode='sparse',     # image étiquetée avec un seul entier correspondant à la classe de l’image
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        shuffle=True
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=X_test,
        directory=images_dir_train,
        x_col='image_name',
        y_col='prdtypecode',
        class_mode='sparse',    # image étiquetée avec un seul entier correspondant à la classe de l’image
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        shuffle=False
    )

    # Chargement du modèle pré-entraîné

    model = keras.models.load_model(f1_score_valid_model_path)
    
    # Prédiction sur l'ensemble de validation
    y_pred_proba = model.predict(valid_generator)
    y_pred_class = np.argmax(y_pred_proba, axis=1).astype(int)
    y_true = valid_generator.classes
    
    # fonction pour calculer le F1-score
    def f1_macro(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')

    # fonction pour calculer le F1-score pondéré
    def f1_weighted(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')
    
    f1_score_macro = f1_macro(y_true, y_pred_class)
    f1_weighted_score = f1_weighted(y_true, y_pred_class)
    classification_rep = classification_report(y_true, y_pred_class)
    
    print("F1-score macro :", f1_score_macro)
    print("F1-score pondéré :", f1_weighted_score)
    print("Rapport de classification :\n", classification_rep)
    
    return f1_weighted_score


# Charger le DataFrame depuis le fichier CSV
#df_test = pd.read_csv('/home/workspace/API/API_rakuten/BDD/dataset/valid_data/df_cleaned.csv')
#df_test['text'].fillna("''", inplace=True)
#f1_weighted = f1_img_test(df_test)
#print("F1-score pondéré final:", f1_weighted)


