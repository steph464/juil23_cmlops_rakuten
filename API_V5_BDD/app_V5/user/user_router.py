from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from passlib.context import CryptContext
from user.user_db import load_users
from sqlalchemy import create_engine, Column, String, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base



router = APIRouter()

# Security configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBasic()

# Spécifier les informations de connexion à la base de données PostgreSQL
db_user = 'postgres'
db_password = 'Process'
db_host = 'localhost'
db_port = '5432'
db_name = 'api_rakuten'
table_name = 'users'

# Configuration de la connexion à la base de données
DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

Base = declarative_base()
metadata = MetaData()

users_table = Table(
    table_name,
    metadata,
    Column("id", String, primary_key=True),
    Column("username", String),
    Column("hashed_password", String),
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Function to authenticate user credentials
def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    users = load_users()
    username = credentials.username
    if username in users and pwd_context.verify(credentials.password, users[username]['hashed_password']):
        return username
    raise HTTPException(
        status_code=401,
        detail="Authentication failed",
        headers={"WWW-Authenticate": "Basic"},
    )

# Endpoint to check if the API is functional
@router.get("/ping", name='Functionality test')
def ping():
    return {"message": "API is functional"}


@router.get("/user")
def current_user(username: str = Depends(authenticate_user)):
    return "Hello {}".format(username)



@router.post("/add_user")
def create_user(username: str, password: str, current_user: str = Depends(authenticate_user)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut ajouter un utilisateur.")
    
    # Check if the username already exists in the database
    session = SessionLocal()
    existing_user = session.execute(users_table.select().where(users_table.c.username == username)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="L'utilisateur existe déjà.")
    
    hashed_password = pwd_context.hash(password)
    
    # Insert new user into the database
    session = SessionLocal()
    session.execute(users_table.insert().values(username=username, hashed_password=hashed_password))
    session.commit()

    return {"message": "Utilisateur ajouté avec succès."}


@router.put("/update_user")
def update_user_password(username: str, new_password: str, current_user: str = Depends(authenticate_user)):
    if current_user != "admin" and current_user != username:
        raise HTTPException(status_code=403, detail="Vous n'avez pas l'autorisation de mettre à jour ce mot de passe.")

    hashed_password = pwd_context.hash(new_password)

    # Update user password in the database
    session = SessionLocal()
    session.execute(
        users_table.update().where(users_table.c.username == username).values(hashed_password=hashed_password)
    )
    session.commit()

    return {"message": "Mot de passe de l'utilisateur mis à jour avec succès."}


@router.delete("/delete_user")
def delete_user_route(username: str, current_user: str = Depends(authenticate_user)):
    if current_user != "admin" and current_user != username:
        raise HTTPException(status_code=403, detail="Vous n'avez pas l'autorisation de supprimer cet utilisateur.")

    # Delete user from the database
    session = SessionLocal()
    session.execute(users_table.delete().where(users_table.c.username == username))
    session.commit()

    return {"message": f"Utilisateur '{username}' supprimé avec succès."}


# Endpoint pour afficher la liste des utilisateurs
@router.get("/list_users", name="Liste des utilisateurs")
def list_users(current_user: str = Depends(authenticate_user)):
    users = load_users()  # Chargez les utilisateurs depuis le fichier JSON

    # Vérifiez que seul l'administrateur peut accéder à cette liste
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à la liste des utilisateurs.")

    # Renvoyez la liste des utilisateurs
    return {"users": users}
