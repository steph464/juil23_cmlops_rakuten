from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from passlib.context import CryptContext
from user.user_db import load_users, save_users, add_user, update_user, delete_user

router = APIRouter()

# Security configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBasic()

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


# Route pour créer un nouvel utilisateur
@router.post("/add_user")
def create_user(username: str, password: str, current_user: str = Depends(authenticate_user)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut ajouter un utilisateur.")

    result = add_user(current_user, username, password)
    return result

# Route pour mettre à jour le mot de passe de l'utilisateur
@router.put("/update_user")
def update_user_password(username: str, new_password: str, current_user: str = Depends(authenticate_user)):
    if current_user != "admin" and current_user != username:
        raise HTTPException(status_code=403, detail="Vous n'avez pas l'autorisation de mettre à jour ce mot de passe.")

    result = update_user(current_user, username, new_password)
    return result

# Route pour supprimer un utilisateur
@router.delete("/delete_user")
def delete_user_route(username: str, current_user: str = Depends(authenticate_user)):
    if current_user != "admin" and current_user != username:
        raise HTTPException(status_code=403, detail="Vous n'avez pas l'autorisation de supprimer cet utilisateur.")

    result = delete_user(username, current_user)
    return result


# Endpoint pour afficher la liste des utilisateurs
@router.get("/list_users", name="Liste des utilisateurs")
def list_users(current_user: str = Depends(authenticate_user)):
    users = load_users()  # Chargez les utilisateurs depuis le fichier JSON

    # Vérifiez que seul l'administrateur peut accéder à cette liste
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à la liste des utilisateurs.")

    # Renvoyez la liste des utilisateurs
    return {"users": users}

