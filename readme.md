# juil23_cmlops_rakuten

## Projet Challenge Rakuten France Multimodal Product Data Classification

## Description du projet

Ce projet vise à prédire le code type de produits (tel que défini dans le catalogue Rakuten France) à partir d'une description texte et d'une image. Il s'agit d'un problème de classification multimodale, où nous utilisons à la fois des données textuelles et visuelles pour prédire la catégorie d'un produit.

## Structure du projet

Le projet est organisé comme suit :

- `app_V4/` : Ce dossier contient le code principal de l'application.
  - `operations/` : Contient les endpoints pour récupérer l'identifiant unique et valider la saisie
  - `predict/` : Contient le code pour faire des prédictions
  - `user/` : Contient le code pour gérer les utilisateurs 
  - `main_router.py` : Le routeur principal de l'application
  - `Dockerfile.app` : Fichier Docker pour la construction de l'image de l'application
  - `main_router.py` : Routeur principal de l'application

 - `BDD/` : Contient les fichiers liés à la base de données
  - `df.ipynb` : Notebook Jupyter concernant la validation des données à partir du modèle bimodal
  - `requirements.txt` : Dépendances 
  - `dataset/` : 
    - `valid_data/` : Contient les fichiers .csv validés
    - `retrain_data/` : Données pour le réentraînement
  - `retrain_text_model/` : Modèles pour le réentraînement du texte
    - `saved_monitoring_journal_retrain/` : sauvegardes des journaux de surveillance après ré-entraînements
    - `saved_txt-retrain_models/` : sauvegardes des rapports de classification après ré-entraînement

- `data/` : Ce dossier contient les données du projet (Y_train_CVw08PX.csv, X_test_update.csv). Notez que certains fichiers volumineux ne sont pas inclus en raison de leur taille

- `models/` : Notez que les fichiers .h5 étant trop volumineux, ils ne sont pas inclus en raison de leur taille

- `notebooks/` : Ce dossier contient les notebooks Jupyter pour l'analyse des données, le prétraitement du texte, la modélisation et la prédiction

- `report/` : Ce dossier contient le cahier des charges du projet

- `test_ci/` : Contient les tests unitaires et un dossier `image_test` avec quelques images pour tester

## Comment ça marche

1. **Prétraitement des données** : Les descriptions de texte sont nettoyées et normalisées. Les images sont redimensionnées et normalisées

2. **Extraction des caractéristiques** : Les caractéristiques sont extraites à partir des descriptions de texte et des images à l'aide de techniques d'apprentissage automatique

3. **Entraînement du modèle** : Un modèle de classification est formé en utilisant les caractéristiques extraites

4. **Prédiction** : Le modèle formé est utilisé pour prédire le code type de produits sur les nouvelles données

## Installation

1. Assurez-vous d'avoir Docker installé sur votre machine

2. Clonez ce dépôt sur votre machine locale

3. Exécutez le docker-compose pour construire et lancer les containers :

-  docker-compose up

4. Pour lancer les tests, utilisez la commande :

- git push
