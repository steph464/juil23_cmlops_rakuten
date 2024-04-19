import pandas as pd
import numpy as np
import os
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.applications import Xception
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, classification_report
from keras.models import load_model, Model


def retrain_img(dataframe):
    
    # Définition des répertoires et des chemins
    images_dir_train = '/home/workspace/API/BDD/RAKUTEN/images/image_train/'
    path_output_models = '/home/workspace/API/BDD/modeles/image/img_saved_retrain_models/'
    
    
    # Chargement des données
    df = dataframe

    # Conversion de la colonne 'prdtypecode' en string
    df['prdtypecode'] = df['prdtypecode'].astype(str)
    
    # Séparation des données en ensembles d'entraînement et de validation
    X_train, X_test = train_test_split(df, train_size=0.8, random_state=1234)

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
    
    # Création du modèle Xception
    xception = Xception(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

    for layer in xception.layers:
        layer.trainable = False

    # Déblocage des 70 couches du modèle xception
    for layer in xception.layers[-70:]:
        layer.trainable = True

    model = Sequential()
    model.add(xception)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=27, activation='softmax'))

    # Compilation du modèle
    learning_rate = 0.0004
    model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=Adam(learning_rate=learning_rate),
                metrics=['accuracy']
                )

    # train_generator.n pour obtenir le nombre total d'images disponibles dans le générateur pour l'ensemble d'entraînement
    step_size_Train = train_generator.n // train_generator.batch_size
    step_size_Valid = valid_generator.n // valid_generator.batch_size
    
    # Sauvegarde des meilleurs poids du modèle au cours de l'entraînement
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = f"Xception_{current_datetime}.h5"  # Nouveau nom du modèle avec la date et l'heure
    model_path = os.path.join("/home/workspace/API/BDD/modeles/image/img_saved_retrain_models", model_name)
    
    # Sauvegarde des meilleurs poids du modèle au cours de l'entraînement :
    model_checkpoint = ModelCheckpoint(filepath=model_path,
                                        monitor='val_accuracy',
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=False,
                                        mode='max',
                                        save_freq='epoch')

    # Sauvegarde des poids (.h5)
    weights_checkpoint = ModelCheckpoint(filepath=path_output_models + "weights_xception.h5",
                                        verbose=1, 
                                        save_best_only=True,
                                        save_weights_only=True,  # True pour sauvegarder uniquement les poids
                                        mode='max',
                                        save_freq='epoch')

    # Réduction automatiquement le learning rate
    early_stopping = EarlyStopping(monitor='val_loss',
                                    min_delta=0,
                                    patience=2,
                                    restore_best_weights=True,
                                    verbose=1,
                                    mode='min')

    # Arrêt de l'entraînement si le modèle n'évolue plus
    lr_plateau = ReduceLROnPlateau(patience=2,
                                    verbose=1)

    callbacks = [model_checkpoint, weights_checkpoint, early_stopping, lr_plateau]
    
    # Entrainement du modèle
    history = model.fit(
                    train_generator,
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    validation_data=valid_generator,
                    validation_steps=valid_generator.samples // valid_generator.batch_size,
                    epochs=25,
                    callbacks=[model_checkpoint, weights_checkpoint, early_stopping, lr_plateau]
                    )
    
    # Évaluation du modèle
    #valid_score = model.evaluate(valid_generator)
    #print("Model metrics names:", model.metrics_names)
    #print("Accuracy: {:.2f}%".format(valid_score[1] * 100))
    #print("Loss: ", valid_score[0])
    #sc = valid_score[0]
    
    # Prédiction sur l'ensemble de validation
    y_pred_proba = model.predict(valid_generator)
    y_pred_class = np.argmax(y_pred_proba, axis=1).astype(int)
    y_true = valid_generator.classes
    
    # Métriques de classification
    f1_macro = f1_score(y_true, y_pred_class, average='macro')
    f1_micro = f1_score(y_true, y_pred_class, average='micro')
    f1_weighted = f1_score(y_true, y_pred_class, average='weighted')
    classification_rep = classification_report(y_true, y_pred_class)
    
        
    #print("F1-score macro :", f1_score_macro)
    #print("F1-score pondéré :", f1_weighted_score)
    #print("Rapport de classification :\n", classification_rep)
    
    # Enregistrement des résultats dans un fichier avec la date dans le nom
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = f"rapport_classification_{current_datetime}.txt"
    output_dir = os.path.join("/home/workspace/API/BDD/modeles/image/img_saved_retrain_models")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Obtenir la date et l'heure actuelles
    with open(os.path.join(output_dir, output_file), "a") as file:
        file.write(f"Date : {current_datetime}\n")
        file.write(f"Modèle : {model_name}\n")
        file.write("Historique :\n")
        file.write(str(history.history))
        file.write("\n")
        file.write(f"F1-score : {f1_macro}\n")
        file.write(f"F1-score pondéré : {f1_weighted}\n")
        file.write("Rapport de classification :\n")
        file.write(classification_rep)
        file.write("\n")
        file.flush()
    
    return f1_weighted

# Charger le DataFrame depuis le fichier CSV
#df_test = pd.read_csv('/home/workspace/API/BDD/df_cleaned.csv')
#df_test['text'].fillna("''", inplace=True)
#f1_weighted = f1_img_test(df_test)
#print("F1-score pondéré final:", f1_weighted)


