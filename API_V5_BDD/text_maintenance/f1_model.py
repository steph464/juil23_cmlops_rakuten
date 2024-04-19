import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.metrics import classification_report

#chemin du modèle pour calcul du f1_weighted score
f1_score_valid_model= "/home/workspace/API/API_rakuten/models/text_models/text_valid_models/conv1D_valid_model.h5"

def f1_text_test(dataframe):
    # Chargement des données
    df = dataframe
    
        
    # Remplissage des valeurs manquantes dans la colonne 'text'
    df['text'].fillna('', inplace=True)
    
    # Séparation des données en ensembles d'entraînement et de test
    X = df['text']
    y = pd.DataFrame(df['prdtypecode'])
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
    
    #Chargez le modèle de classification textuelle
    model = load_model(f1_score_valid_model)
    
    # Calcul du F1-score pondéré sur l'ensemble de test
    y_pred = model.predict(X_test)
    f1_score_macro = f1_score(y_test, y_pred.argmax(axis=1), average='macro')
    f1_weighted = f1_score(y_test, y_pred.argmax(axis=1), average='weighted')
    classification_rep = classification_report(y_test, y_pred.argmax(axis=1))
    
            
    return f1_weighted
    
    
#Charger le DataFrame depuis le fichier CSV
#df_test= pd.read_csv('/home/workspace/API/BDD/df_cleaned.csv')
#df_test['text'].fillna("''", inplace=True)
#result = f1_text_test(df_test)
#print("F1-weighted score:", result)

    
    
    
    
