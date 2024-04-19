import requests
import base64
import os
import sys
import json

# Ajout du chemin du répertoire parent (app) au chemin de recherche des modules Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importation des fonctions delete_user et load_users depuis user_db.py  
from app_V4.user.user_db import delete_user, load_users

# Définition de l'adresse de l'API et du port
api_address = 'localhost'  
api_port = 8005

def test_delete_user(admin_username, admin_password, username_to_delete, expected_status_code): 
    # Charger les utilisateurs depuis la base de données
    users = load_users()

    # Vérifiez si le nom d'utilisateur existe 
    if username_to_delete not in users:
        print(f"/!\ L'utilisateur '{username_to_delete}' n'existe pas dans la base de données.")

        expected_status_code = 403  # Ajustez le code de statut HTTP attendu

    # Authentification de l'administrateur
    auth_header = f'Basic {base64.b64encode(f"{admin_username}:{admin_password}".encode()).decode()}'
    headers = {'Authorization': auth_header}

    # Données pour la suppression de l'utilisateur
    data = {'username': username_to_delete}

    # Test pour l'endpoint "/delete_user"
    delete_user_url = f'http://{api_address}:{api_port}/Users/delete_user?username={username_to_delete}'   
    
    # Effectuez la requête HTTP vers l'API pour supprimer l'utilisateur
    response = requests.delete(delete_user_url, headers=headers, json=data)

    # Obtenez le code d'état HTTP de la réponse
    actual_status_code = response.status_code


    output = f'''
    ============================
        Delete User Test
    ============================

    request done at "Users/delete_user/{username_to_delete}"     
    | admin_username="{admin_username}"
    | admin_password="{admin_password}"
    
    expected result = {expected_status_code}

    actual result = {actual_status_code}

    ==>  {"SUCCESS" if actual_status_code == expected_status_code else "FAILURE"}
    '''

    # Affichage des résultats
    print(output)

    # Impression dans un fichier (si l'environnement LOG est défini à 1)
    if os.environ.get('LOG') == '1':
        with open('api_test.log', 'a') as file:
            file.write(output)

# Test pour supprimer un utilisateur avec l'administrateur (statut HTTP 200 attendu)
test_delete_user('admin', 'boss', 'bob', 200)

# Test pour supprimer un utilisateur en tant qu'utilisateur non administrateur (statut HTTP 403 attendu : accès refusé)
test_delete_user('stephane', 'sport', 'alice', 403)
