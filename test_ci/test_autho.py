import os
import requests
import base64

# Définition de l'adresse de l'API et du port
api_address = 'localhost'  
api_port = 8005

def test_authentication_for_predict_text(username: str, password: str, expected_status_code: int):
    auth_header = f'Basic {base64.b64encode(f"{username}:{password}".encode()).decode()}'
    headers = {
        'Authorization': auth_header,
        'accept': 'application/json',
    }

    # Endpoint "/predict_text_class"
    predict_text_url = f'http://{api_address}:{api_port}/Predictions/predict_text_class'
    
    # Données de texte de test
    designation: str = "AstralCom 1 CV Mono de DAB - Surpresseur piscine"
    description: str = "Tension (V) : Monophasé - 230 V  Puissance : 1 CV  Turbine (nombre) : 4  Compatible traitement au sel : Oui"

    # Paramètres de la requête
    params = {
        'designation': designation,
        'description': description,
    }

    r = requests.post(predict_text_url, headers=headers, params=params)
    output = f'''
    ============================
        Autorisation test for /predict_text_class    
    ============================

    request done at "/Predictions/predict_text_class"
    | username="{username}"
    | password="{password}"
    
    expected result = {expected_status_code}
    actual result = {r.status_code}

    ==>  {"SUCCESS" if r.status_code == expected_status_code else "FAILURE"}
    '''

    # Affichage des résultats
    print(output)

    # Impression dans un fichier (si l'environnement LOG est défini à 1)
    if os.environ.get('LOG') == '1':
        with open('api_test.log', 'a') as file:
            file.write(output)

# Test pour les utilisateurs existants
test_authentication_for_predict_text('david', 'chocolat', 200)

# Test pour un utilisateur avec mot de passe incorrect
test_authentication_for_predict_text('clementine', 'mandarine', 401)  
