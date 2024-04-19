# predict_router.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
from pydantic import BaseModel
from fastapi import UploadFile, Depends
from user.user_router import authenticate_user  # Importez votre fonction d'authentification ici
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Boolean
from typing import Dict

operations = APIRouter()

# Définissez la classe de réponse JSON
class Prediction(BaseModel):
    filename: str
    content_type: str
    predictions: List[dict] = []


###################################" chemins des modèles et dataframe ###################################
    
# chemin du modèle texte valide    
valid_text_model_path = "/home/workspace/API/API_BDD/models/text_models/text_valid_models/conv1D_valid_model.h5"

# chemin du dataset de sauvegarde Texte.data valid
text_data_path = "/home/workspace/API/API_BDD/BDD/dataset/valid_data/Texte_data.csv"

                        ########################################################
                        
#chemin du modèle image valide
valid_img_model_path="/home/workspace/API/API_BDD/models/img_models/img_valid_models/Xception_valid_model.h5"

# chemin de sauvegarde des images prédites
img_data_path="/home/workspace/API/API_BDD/BDD/saved_img_data/Xception_img"

# chemin du dataset de sauvegarde de img_data valid
img_dataset_path="/home/workspace/API/API_BDD/BDD/dataset/valid_data/img_data.csv"

                        ##########################################################

# chemin de sauvegarde de bimod_data, dataset du traitement bimodal
bimod_data_path= "/home/workspace/API/API_BDD/BDD/dataset/valid_data/bimod_data.csv"

# chemin de sauvegarde des images du traitement bimodal
bimod_img_path="/home/workspace/API/API_BDD/BDD/saved_img_data/bimod_img"

########################################################################################################
########################################################################################################

# Spécifier les informations de connexion à la base de données PostgreSQL
db_user = 'postgres'
db_password = 'Process'
db_host = 'localhost'
db_port = '5432'
db_name = 'api_rakuten'
table_name = 'texte_data'
table_img_name = 'img_data'
table_name_bimod = 'bimod_data'

# Configuration de la connexion à la base de données
DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Declare a SQLAlchemy Base model
Base = declarative_base()

# Define the TexteData model for the texte_data table
class TexteData(Base):
    __tablename__ = 'texte_data'

    id = Column(Integer, primary_key=True, index=True)
    unique_id = Column(String, index=True)
    is_correct = Column(Boolean)

# Create an instance of the engine and bind it to the Base
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        


#######################################################################################################
@operations.get("/recover_texte_data_by_id/{unique_id}")
async def get_text_data_by_id(unique_id: str, current_user: str = Depends(authenticate_user), db: SessionLocal = Depends(get_db)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionnalité.")

    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}', connect_args={'options': '-c search_path=public'})

    # Lire les données de la table dans un DataFrame
    query = f'SELECT * FROM "{table_name}";'
    texte_data_psql= pd.read_sql_query(query, engine)
    
    
    # Recherchez la ligne correspondant au numéro unique donné
    data_row = texte_data_psql[texte_data_psql['unique_id'] == unique_id]
    
    if data_row.empty:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
    # Remplacer les NaN par des valeurs par défaut (par exemple, une chaîne vide)
    data_row = data_row.fillna('')
    
    # Convertir la ligne en un dictionnaire JSON
    data_dict = data_row.iloc[0].to_dict()
    
    return data_dict
    
#######################################################################################################

@operations.put("/update_texte_prediction_status")
async def update_texte_prediction_status(unique_id: str, is_correct: bool, current_user: str = Depends(authenticate_user), db: SessionLocal = Depends(get_db)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionalité.")
    
    # Charger le DataFrame existant (Texte_data)
    Texte_data = pd.read_csv(text_data_path)
    
    # Recherchez la ligne correspondant au numéro unique donné
    data_row_index = Texte_data[Texte_data['unique_id'] == unique_id].index
    
    if len(data_row_index) == 0:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
    # Mettre à jour le statut de prédiction
    Texte_data.at[data_row_index[0], 'is_correct'] = is_correct
    
    # Sauvegarder le DataFrame mis à jour
    Texte_data.to_csv(text_data_path, index=False)
    
    # Créer une connexion à la base de données
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    
    # Insérer le DataFrame dans la base de données PostgreSQL
    Texte_data.to_sql('texte_data', engine, index=False, if_exists='replace')

    return {"message": f"Statut de prédiction mis à jour pour l'ID unique {unique_id} : is_correct={is_correct}"}

#########################################################################################################

# Endpoint pour récupérer des données en fonction du numéro unique pour le traitement de l'image
@operations.get("/recover_image_data_by_id/{unique_id}")
async def get_img_data_by_id(unique_id: str, current_user: str = Depends(authenticate_user), db: SessionLocal = Depends(get_db)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionalité.")
    
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}', connect_args={'options': '-c search_path=public'})

    # Lire les données de la table dans un DataFrame
    query = f'SELECT * FROM "{table_img_name}";'
    img_data_psql= pd.read_sql_query(query, engine)
    
    # Recherchez la ligne correspondant au numéro unique donné
    data_row = img_data_psql[img_data_psql['unique_id'] == unique_id]
    
    if data_row.empty:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
    # Remplacer les NaN par des valeurs par défaut (par exemple, une chaîne vide)
    data_row = data_row.fillna('')
    
    # Convertir la ligne en un dictionnaire JSON
    data_dict = data_row.iloc[0].to_dict()
    
    return data_dict
    
#########################################################################################################


@operations.put("/update_image_prediction_status")
async def update_image_prediction_status(unique_id: str, is_correct: bool, current_user: str = Depends(authenticate_user), db: SessionLocal = Depends(get_db)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionalité.")
    
    # Charger le DataFrame existant (img_data)
    img_data = pd.read_csv(img_dataset_path)
    
    # Recherchez la ligne correspondant au numéro unique donné
    data_row_index = img_data[img_data['unique_id'] == unique_id].index
    
    if len(data_row_index) == 0:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
    # Mettre à jour le statut de prédiction
    img_data.at[data_row_index[0], 'is_correct'] = is_correct
    
    # Sauvegarder le DataFrame mis à jour
    img_data.to_csv(img_dataset_path, index=False)
    
    
    # Créer une connexion à la base de données
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    
    # Insérer le DataFrame dans la base de données PostgreSQL
    img_data.to_sql(table_img_name, engine, index=False, if_exists='replace')

    return {"message": f"Statut de prédiction mis à jour pour l'ID unique {unique_id} : is_correct={is_correct}"}

#########################################################################################################


# Endpoint pour récupérer des données en fonction du numéro unique
@operations.get("/recover_bimod_data_by_id/{unique_id}")
async def get_bimod_data_by_id(unique_id: str, current_user: str = Depends(authenticate_user),db: SessionLocal = Depends(get_db)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionalité.") 
    
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}', connect_args={'options': '-c search_path=public'})

    # Lire les données de la table dans un DataFrame
    query = f'SELECT * FROM "{table_name_bimod}";'
    bimod_data_psql= pd.read_sql_query(query, engine)
    
    # Recherchez la ligne correspondant au numéro unique donné
    data_row = bimod_data_psql[bimod_data_psql['unique_id'] == unique_id]
    
    if data_row.empty:
        raise HTTPException(status_code=404, detail="Données non trouvées")
    
    # Remplacer les NaN par des valeurs par défaut (par exemple, une chaîne vide)
    data_row = data_row.fillna('')
    
    # Convertir la ligne en un dictionnaire JSON
    data_dict = data_row.iloc[0].to_dict()
    
    return data_dict
    

########################################################################################################
