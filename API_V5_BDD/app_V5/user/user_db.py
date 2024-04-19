
from passlib.context import CryptContext
import json
from sqlalchemy import create_engine, Column, String, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


# Configuration de la sécurité
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Spécifier les informations de connexion à la base de données PostgreSQL
db_user = 'postgres'
db_password = 'Process'
db_host = 'localhost'
db_port = '5432'
db_name = 'api_rakuten'
table_name = 'users'

# Configuration de la connexion à la base de données
DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
metadata = MetaData()

users_table = Table(
    table_name,
    metadata,
    Column("id", String, primary_key=True),
    Column("username", String),
    Column("hashed_password", String),
)

def load_users():
    session = SessionLocal()
    result = session.execute(users_table.select())
    return {row[1]: {"hashed_password": row[2]} for row in result}

# Fonction pour sauvegarder les données des utilisateurs dans postgresql: api_rakuten
def save_users(users_data):
    session = SessionLocal()
    for username, data in users_data.items():
        session.execute(
            users_table.insert().values(username=username, hashed_password=data["hashed_password"])
        )
    session.commit()
    

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

    # Enregistrez les données des utilisateurs mises à jour
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

