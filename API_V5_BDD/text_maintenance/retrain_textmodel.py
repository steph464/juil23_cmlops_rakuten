
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
import shutil

# chemin d'enregistrement du meilleur modèle pendant l'entrainement
retrain_text_bestmodel_path = "/home/workspace/API/API_BDD/BDD/retrain_text_model/saved_txt-retrain_models"

#chemin d'enregistrement du journal de réentrainement
journal_retrain_path="/home/workspace/API/API_BDD/BDD/retrain_text_model/saved_txt-retrain_models"

# chemin de copie du bets model réentrainé pour utilisatuion par la route de prédiction
valid_text_path="/home/workspace/API/API_BDD/models/text_models/text_valid_models"


def retrain_text(dataframe):
    # Chargement des données
    df = dataframe
    
        
    # Remplissage des valeurs manquantes dans la colonne 'text'
    df['text'].fillna('', inplace=True)
    
    # Séparation des données en ensembles d'entraînement et de test
    X = df['text']
    y = pd.DataFrame(df['prdtypecode']).astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tokenisation du texte d'entraînement
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Comptage du nombre de mots uniques
    word_size = len(tokenizer.word_index) + 1

    # Longueur maximale de chaque séquence de mots
    maxlen = 400

    # Représentation par un vecteur de 100 dimensions
    embedding_dim = 100
    
    # Padding des séquences de mots
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    
    # Définition du modèle
    model = keras.Sequential()
    model.add(layers.Embedding(word_size, embedding_dim, input_length=maxlen))
    model.add(layers.SpatialDropout1D(0.2))
    model.add(layers.Conv1D(64, 2, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(27, activation='softmax'))

    # Compilation du modèle
    lr = 0.001
    model.compile(optimizer=Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Sauvegarde des meilleurs poids du modèle au cours de l'entraînement
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = f"conv1D_{current_datetime}.h5"  # Nouveau nom du modèle avec la date et l'heure
    model_path = os.path.join(retrain_text_bestmodel_path, model_name)  # Chemin complet du modèle

    
    checkpoint = ModelCheckpoint(filepath=model_path,
                                monitor='val_accuracy',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False,
                                mode='max',
                                save_freq='epoch')

    
    # Réduction automatique du learning rate
    early = EarlyStopping(monitor='val_loss',
                        min_delta=0,
                        patience=7,  # patience=10
                        restore_best_weights=True,
                        verbose=1,
                        mode='min')

    # Arrêt de l'entraînement si le modèle n'évolue plus
    lr_reduce = ReduceLROnPlateau(patience=1, #patience=2
                                verbose=1)

    callbacks = [checkpoint, early, lr_reduce]
    
    # Entraînement du modèle
    history = model.fit(X_train, y_train.values,
                    batch_size=200,
                    epochs=20,
                    validation_data=(X_test, y_test.values),
                    callbacks=callbacks)
    
    
    # Calcul du F1-score pondéré sur l'ensemble de test
    y_pred = model.predict(X_test)
    f1_score_macro = f1_score(y_test, y_pred.argmax(axis=1), average='macro')
    f1_weighted = f1_score(y_test, y_pred.argmax(axis=1), average='weighted')
    classification_rep = classification_report(y_test, y_pred.argmax(axis=1))
    
    # Enregistrement des résultats dans un fichier avec la date dans le nom
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = f"rapport_classification_{current_datetime}.txt"
    output_dir = os.path.join(journal_retrain_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, output_file), "a") as file:
        file.write(f"Date : {current_datetime}\n")
        file.write(f"Modèle : {model_name}\n")
        file.write("Historique :\n")
        file.write(str(history.history))
        file.write("\n")
        file.write(f"F1-score : {f1_score_macro}\n")
        file.write(f"F1-score pondéré : {f1_weighted}\n")
        file.write("Rapport de classification :\n")
        file.write(classification_rep)
        file.write("\n")
        file.flush()
    

    ################################################################################################
    # Copier le modèle sauvegardé vers le répertoire de validation
    source_model_path = model_path  # Chemin du modèle sauvegardé
    target_model_path = os.path.join(valid_text_path, "conv1D_valid_model.h5")
    
    # Chemin pour sauvegarder le modèle précédent
    previous_model_path = os.path.join("/home/workspace/API/API_rakuten/models/text_models/text_saved_previous_models", f"conv1D_previous_model_{current_datetime}.h5")

    # Si le modèle de validation existe déjà, sauvegardez-le avant de le supprimer
    if os.path.exists(target_model_path):
        # Copier le modèle vers le répertoire de sauvegarde
        shutil.copy(target_model_path, previous_model_path)
        # Supprimer le modèle de validation actuel
        os.remove(target_model_path)

    # Copie le nouveau modèle vers le répertoire de validation
    shutil.copy(source_model_path, target_model_path)


