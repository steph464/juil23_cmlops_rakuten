import requests
import base64
import os
import sys
import json

# Ajout du chemin du répertoire parent (app) au chemin de recherche des modules Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importation des fonctions delete_user et load_users depuis user_db.py
from app_V4.user.user_db import add_user, load_users

# Définition de l'adresse de l'API et du port
api_address = 'localhost'  
api_port = 8005

def test_create_user(admin_username, admin_password, new_username, new_password, expected_status_code):
    # Charger les utilisateurs depuis la base de données
    users = load_users()

    # Vérifiez si le nom d'utilisateur existe
    if new_username in users:
        print(f"/!\ L'utilisateur '{new_username}' existe déjà dans la base de données.")
        return

    # Vérifier si l'utilisateur est l'administrateur
    if admin_username != "admin":
        print(f"/!\ Seul l'administrateur est autorisé à créer de nouveaux utilisateurs.")
        return

    # Authentification de l'administrateur
    auth_header = f'Basic {base64.b64encode(f"{admin_username}:{admin_password}".encode()).decode()}'
    headers = {'Authorization': auth_header}

    # Données pour la création de l'utilisateur
    data = {'username': new_username, 'password': new_password}

    # Test pour l'endpoint "Users/add_user"
    add_user_url = f'http://{api_address}:{api_port}/Users/add_user?username={new_username}&password={new_password}' 
    
    r = requests.post(add_user_url, headers=headers, json=data)   
    
    output = f'''
    ============================
        Create User Test
    ============================

    request done at "/Users/add_user"
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
test_create_user('admin', 'boss', 'bob', 'eponge', 200)

# Test pour créer un nouvel utilisateur en tant qu'utilisateur non administrateur (statut HTTP 403 attendu)
test_create_user('david', 'chocolat', 'jo', 'musique', 403)
