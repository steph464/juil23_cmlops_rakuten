import os
import requests
import base64

# Définition de l'adresse de l'API et du port
api_address = 'localhost' 
api_port = 8005

def test_predict_text(username: str, password: str, expected_status_code: int):
    auth_header = f'Basic {base64.b64encode(f"{username}:{password}".encode()).decode()}'
    headers = {
        'Authorization': auth_header,
        'accept': 'application/json',
    }

    # Endpoint "/predict_text_class"
    predict_text_url = f'http://{api_address}:{api_port}/Predictions/predict_text_class'
    
    # Données de texte de test
    designation: str = "Robot de piscine électrique"
    description: str = "<p>Ce robot de piscine d&#39;un design innovant et élégant apportera un nettoyage efficace et rapide."
    real_code= 2583
    
    # Paramètres de la requête
    params = {
        'designation': designation,
        'description': description,
    }

    response = requests.post(predict_text_url, headers=headers, params=params)
    
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
    
    output = f'''
====================================================================
    Predictions test using CONV1D BASED MODEL - results
====================================================================

request done at "predict_text_class"
| username = '{username}'
| password = '{password}'
| designation = '{designation}'
| description = '{ description}'

expected result code = {expected_status_code}
expected Predicted class code =  {real_code}

actual result code = {status_code}
actual Predicted class code = {predicted_code}  
==> Code result = {test_status}
==> Prediction result = {prediction_status}
'''

    
    
    # Affichage des résultats
    print(output)

    # Impression dans un fichier (si l'environnement LOG est défini à 1)
    if os.environ.get('LOG') == '1':
        with open('api_test.log', 'a') as file:
            file.write(output)

# Test pour les utilisateurs existants
test_predict_text('admin', 'boss', 200)

