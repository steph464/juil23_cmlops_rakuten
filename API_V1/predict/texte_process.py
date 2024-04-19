
import re
import pandas as pd     ##############"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import keras

designation = "It - Bobble Head Pop N° 539 - Beverly With Key.."
description = "NaN"

data = {'designation': [designation],
        'description': [description]}

df1 = pd.DataFrame(data)
df1 = df1.astype(str)

from nltk.corpus import stopwords

stop_words = stopwords.words('french') + stopwords.words('english') + stopwords.words('german') +[ \
                'plus', 'peut', 'tout', 'etre', 'sans', 'dont', 'aussi',  \
                  'comme', 'meme', 'bien','leurs', 'elles', 'cette','celui',   \
                  'ainsi', 'encore', 'alors', 'toujours', 'toute','deux', 'nouveau',   \
                  'peu', 'car', 'autre', 'jusqu', 'quand', 'ici', 'ceux', 'enfin',  \
                  'jamais', 'autant', 'tant', 'avoir', 'moin', 'celle', 'tous',   \
                  'contre', 'pourtant', 'quelque', 'toutes', 'surtout', 'cet',  \
                  'comment', 'rien', 'avant', 'doit', 'autre', 'depuis', 'moins',  \
                  'tre', 'souvent', 'etait', 'pouvoir', 'apre', 'non', 'ver', 'quel',   \
                  'pourquoi', 'certain', 'fait', 'faire', 'sou', 'donc', 'trop',  \
                  'quelques', 'parfois', 'tres', 'donc', 'dire', 'eacute', 'egrave',  \
                  'rsquo', 'agrave', 'ecirc', 'nbsp', 'acirc', 'apres', 'autres',  \
                  'ocirc', 'entre', 'sous', 'quelle', 'NaN', 'nan']



def CreateTextANDcleaning(data):

  df = data
  #valeurs MANQUANTES
  df['designation'] = df['designation'].astype('string')
  df['description'] = df['description'].astype('string')

  #create text
  df['text']=""
  for i in range(df.shape[0]):
    df['text'][i] = create_text(df['designation'][i], df['description'][i])

  df['text'] = df['text'].str.split()

  df['text'] = df['text'].apply(lambda x: unique_description(x))
  df['text'] = df['text'].apply(lambda x: " ".join(x))

  df['text'] = df['text'].apply(lambda text : lower_case(text))
  df['text'] = df['text'].apply(lambda text : remove_accent(text))

  df['text'] = df['text'].apply(lambda text : remove_htmltags(text))
  df['text'] = df['text'].apply(lambda text : keeping_essentiel(text))

  # Initialiser la variable des mots vides
  df['text'] = df['text'].str.split()
  df['text']= df['text'].apply(lambda x: operation(x))

  df['text'] = df['text'].apply(lambda x: " ".join(x))

  return df['text']


def create_text(text1, text2):
    if pd.isna(text2):
        text = text1
    else:
        text = text1 +" "+ text2
    return text

def unique_description(text):
    unique=[text[0]]
    for mot in text:
        if mot not in unique:
            unique.append(mot)
    return unique

def lower_case(text):
    text = text.lower().strip()
    return text

def remove_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')

    return string

def remove_htmltags(text):
    text = re.sub('<[^<]+?>', '',text)
    return text

def keeping_essentiel(text):
    text = re.sub(r"[^a-zA-Z]+", " ", text)
    return text

def operation(x):
    my_list=[]
    for mot in x:
        if (mot not in stop_words and len(mot)>2):
            my_list.append(mot)
    return my_list

