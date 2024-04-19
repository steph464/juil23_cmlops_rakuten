
from passlib.context import CryptContext
import json

# Configuration de la sécurité
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Fonction pour charger les données des utilisateurs depuis le fichier JSON
def load_users():
    try:
        with open("/home/workspace/API/app/user/users.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Fonction pour enregistrer les données des utilisateurs dans le fichier JSON
def save_users(users_data):
    with open("/home/workspace/API/app/user/users.json", "w") as file:
        json.dump(users_data, file, indent=4)

# Charger les données des utilisateurs
users = load_users()

######################################################################################################""
def add_user(admin_username, new_username, password):
    # Vérifiez d'abord si l'utilisateur est un admin
    if admin_username != "admin":
        return {"message": "Seul l'administrateur peut ajouter un utilisateur."}

    # Vérifiez si le nom d'utilisateur existe déjà
    if new_username in users:
        return {"message": "L'utilisateur existe déjà."}

    # Ajoutez le nouvel utilisateur avec le mot de passe haché
    hashed_password = pwd_context.hash(password)
    users[new_username] = {
        "username": new_username,
        "hashed_password": hashed_password,
    }

    # Enregistrez les données des utilisateurs mises à jour dans le fichier JSON
    save_users(users)

    return {"message": "Utilisateur ajouté avec succès."}

############################################################################################################"


def update_user(admin_username, username, new_password):
    # Vérifiez d'abord si l'utilisateur est un admin
    if admin_username != "admin":
        return {"message": "Seul l'administrateur peut mettre à jour un utilisateur."}

    # Vérifiez si le nom d'utilisateur existe
    if username not in users:
        return {"message": "L'utilisateur n'existe pas."}

    # Mettez à jour le mot de passe de l'utilisateur avec le mot de passe haché
    hashed_password = pwd_context.hash(new_password)
    users[username]["hashed_password"] = hashed_password

    # Enregistrez les données des utilisateurs mises à jour dans le fichier JSON
    save_users(users)

    return {"message": "Mot de passe de l'utilisateur mis à jour avec succès."}

########################################################################################################""""""""


def delete_user(username, admin_username):
    
    # Vérifiez si l'utilisateur actuel est l'administrateur ou l'utilisateur que vous souhaitez supprimer
    if admin_username != "admin" and username != admin_username:
        return {"message": "Vous n'avez pas l'autorisation de supprimer cet utilisateur."}

    # Vérifiez si le nom d'utilisateur existe
    if username not in users:
        return {"message": "L'utilisateur n'existe pas."}

    # Supprimez l'utilisateur
    del users[username]

    # Enregistrez les données des utilisateurs mises à jour dans le fichier JSON
    save_users(users)

    return {"message": f"Utilisateur '{username}' supprimé avec succès."}

