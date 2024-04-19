import json
from passlib.context import CryptContext

# Configuration de la sécurité
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Création du profil de l'administrateur
admin_password = 'boss'
hashed_password = pwd_context.hash(admin_password)

admin_profile = {
    "admin": {
        "username": "admin",
        "password": admin_password,
        "hashed_password": hashed_password,
    }
}

# Enregistrez le profil de l'administrateur dans un fichier JSON
with open("users.json", "w") as file:
    json.dump(admin_profile, file, indent=4)
