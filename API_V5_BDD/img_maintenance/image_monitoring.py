import os
import requests
import datetime
import base64
import pandas as pd
import tensorflow as tf
from f1_image import f1_img_test
from retrain_immodel import retrain_img


############################################### donnée de test ##########################################
# Chemin vers l'image à tester
test_image_path = '/home/workspace/API/API_rakuten/img_maintenance/test/image_916824168_product_160284580.jpg'
real_label = "jeux vidéos"
real_code = 40


###########################################" chemin des modèles et data ##################################
# chemin d'enregistrement du meilleur modèle pendant l'entrainement
df_test_path="/home/workspace/API/API_rakuten/BDD/dataset/valid_data/df_cleaned.csv"

# chemin du dataset de réentrainenemnt
df_retrain_path="/home/workspace/API/API_rakuten/BDD/dataset/valid_data/df_cleaned.csv"

# chemin du journal de monitoring
log_directory="/home/workspace/API/API_rakuten/BDD/retrain_img_model/saved_monitoring_journal_retrain"

##########################################################################################################


# Définition de l'adresse de l'API et du port
api_address = 'localhost'  # Remplacez par l'adresse de votre API
api_port = 8000

def test_predict_image(username: str, password: str, expected_status_code: int):
    auth_header = f'Basic {base64.b64encode(f"{username}:{password}".encode()).decode()}'
    headers = {
        'Authorization': auth_header,
        'accept': 'application/json',
    }

    # Endpoint "/predict_image_class"
    predict_image_url = f'http://{api_address}:{api_port}/Predictions/predict_image_class'    

    # Vérifier si le fichier image existe
    if not os.path.exists(test_image_path):
        print(f"Image not found at {test_image_path}")
        return

    # Envoi de la requête POST avec l'image
    with open(test_image_path, 'rb') as image_file:
        files = {'file': (os.path.basename(test_image_path), image_file, 'image/jpeg')}

        response = requests.post(predict_image_url, headers=headers, files=files)

    # Response code status
    status_code = response.status_code

    # Paramètres pour la sortie
    test_status = 'SUCCESS' if status_code == expected_status_code else 'FAILURE'

    # Request response data
    results = response.json()

    predicted_code = results.get("predicted_code")

    # Setting test results
    if status_code == 200:
        prediction_status = 'SUCCESS' if predicted_code == real_code else 'FAILURE'
    else:
        prediction_status = 'FAILURE'

    output_1 = f'''
====================================================================
    Predictions test using Xception IMAGE BASED MODEL - results
====================================================================

request done at "predict_image_class"
| username = '{username}'
| password = '{password}'
| Image path = '{test_image_path}'

expected result code = {expected_status_code}
expected Predicted class code =  {real_code}

actual result code = {status_code}
actual Predicted class code = {predicted_code}  
==> Code result = {test_status}
==> Prediction result = {prediction_status}
'''

    ####################################################################################################
    ######################  test du f1_weighted score sur le dataset ###################################
    f1_target= 0.65
    #Charger le DataFrame depuis le fichier CSV
    df_test= pd.read_csv(df_test_path)
    
    result = f1_img_test(df_test)
    
    f1_error = abs(f1_target- result)
    is_f1_error = (f1_error < 0.15)
    
    output_2 = f'''
====================================================================
    f1_weighted score monitoring - results
====================================================================

request done at "predict_text_class"
| username = '{username}'
| password = '{password}'

f1_weighted score ={ result}
f1_target = {f1_target}
f1_error (must be < 0.15)= {f1_error}
==> model is valid ={is_f1_error}
'''
   
    # Affichage des résultats
    print(output_1)
    print(output_2)
    
    
    #df_retrain = pd.read_csv(df_retrain_path)
    # Si le F1 weighted score est inférieur de 10%
    #if not is_f1_error:
    #    retrain_img(df_retrain)
    
    
    #Fonction pour générer un nom de fichier de journal basé sur la date et l'heure actuelles
    def generate_log_filename():
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'log_{formatted_time}.txt'
        return log_filename

    #Écriture de output_1 et output_2 dans un fichier de journal dans le répertoire spécifié
    log_filename = generate_log_filename()
    log_path = os.path.join(log_directory, log_filename)

    with open(log_path, 'w') as log_file:
        log_file.write(output_1)
        log_file.write(output_2)
        print(f"Log file '{log_path}' created.")
        
        
    
# Test pour les utilisateurs existants
test_predict_image('admin', 'boss', 200)

    