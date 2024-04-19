import os
import requests
import base64

# Définition de l'adresse de l'API et du port
api_address = 'localhost'  
api_port = 8005

def test_authentication(username, password, expected_status_code):
    auth_header = f'Basic {base64.b64encode(f"{username}:{password}".encode()).decode()}'
    headers = {'Authorization': auth_header}

    # Test pour l'endpoint "/user"
    user_url = f'http://{api_address}:{api_port}/Users/user'
    r = requests.get(user_url, headers=headers)
    output = f'''
    ============================
        Authentication test
    ============================

    request done at "/Users/user"
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
test_authentication('david', 'chocolat', 200)
test_authentication('stephane', 'sport', 200)

# Test pour un utilisateur avec mot de passe incorrect
test_authentication('clementine', 'mandarine', 401)  
