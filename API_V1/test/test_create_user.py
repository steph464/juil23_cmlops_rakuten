import requests
import base64
import os

# Définition de l'adresse de l'API et du port
api_address = 'localhost'  # Remplacez par l'adresse de votre API
api_port = 8000

def test_create_user(admin_username, admin_password, new_username, new_password, expected_status_code):
    # Authentification de l'administrateur
    auth_header = f'Basic {base64.b64encode(f"{admin_username}:{admin_password}".encode()).decode()}'
    headers = {'Authorization': auth_header}

    # Données pour la création de l'utilisateur
    data = {'username': new_username, 'password': new_password}

    # Test pour l'endpoint "/add_user"
    add_user_url = f'http://{api_address}:{api_port}/Users/add_user'  # Utilisez "/add_user" au lieu de "/Users/add_user"
    
    
    r = requests.post(add_user_url, headers=headers, data=data)
    
    
    output = f'''
    ============================
        Create User Test
    ============================

    request done at "/add_user"
    | admin_username="{admin_username}"
    | admin_password="{admin_password}"
    | new_username="{new_username}"
    | new_password="{new_password}"
    
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

# Test pour créer un nouvel utilisateur avec l'administrateur (statut HTTP 200 attendu)
test_create_user('admin', 'boss', 'bob', 'tomate', 200)

# Test pour créer un nouvel utilisateur en tant qu'utilisateur non administrateur (statut HTTP 403 attendu)
test_create_user('david', 'chocolat', 'jo', 'musique', 403)
