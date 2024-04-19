import os
import requests
import base64


############################################### donnée de test ##########################################
# Chemin vers l'image à tester
test_image_path = '/home/workspace/API/API_rakuten/img_maintenance/test/image_916824168_product_160284580.jpg'
real_label = "jeux vidéos"
real_code = 40


###########################################" chemin des modèles et data ##################################
# chemin d'enregistrement du meilleur modèle pendant l'entrainement
retrain_text_bestmodel_path = "/home/workspace/API/API_rakuten/BDD/retrain_text_model/saved_txt-retrain_models"

#chemin d'enregistrement du journal de réentrainement
journal_retrain_path="/home/workspace/API/API_rakuten/BDD/retrain_text_model/saved_txt-retrain_models"

# chemin de copie du bets model réentrainé pour utilisatuion par la route de prédiction
valid_text_path="/home/workspace/API/API_rakuten/models/text_models/text_valid_models"

##########################################################################################################


# Définition de l'adresse de l'API et du port
api_address = 'localhost'  # Remplacez par l'adresse de votre API
api_port = 8000

def test_predict_image(username: str, password: str, expected_status_code: int):
    auth_header = f'Basic {base64.b64encode(f"{username}:{password}".encode()).decode()}'
    headers = {
        'Authorization': auth_header,
        'accept': 'application/json',
    }

    # Endpoint "/predict_image_class"
    predict_image_url = f'http://{api_address}:{api_port}/Predictions/predict_image_class'    

    # Vérifier si le fichier image existe
    if not os.path.exists(image_path):
        print(f"Image not found at {test_image_path}")
        return

    # Envoi de la requête POST avec l'image
    with open(test_image_path, 'rb') as image_file:
        files = {'file': (os.path.basename(test_image_path), image_file, 'image/jpeg')}

        response = requests.post(predict_image_url, headers=headers, files=files)

    # Response code status
    status_code = response.status_code

    # Paramètres pour la sortie
    test_status = 'SUCCESS' if status_code == expected_status_code else 'FAILURE'

    # Request response data
    results = response.json()

    predicted_code = results.get("predicted_code")

    # Setting test results
    if status_code == 200:
        prediction_status = 'SUCCESS' if predicted_code == real_code else 'FAILURE'
    else:
        prediction_status = 'FAILURE'

    output = f'''
====================================================================
    Predictions test using Xception IMAGE BASED MODEL - results
====================================================================

request done at "predict_image_class"
| username = '{username}'
| password = '{password}'
| Image path = '{image_path}'

expected result code = {expected_status_code}
expected Predicted class code =  {real_code}

actual result code = {status_code}
actual Predicted class code = {predicted_code}  
==> Code result = {test_status}
==> Prediction result = {prediction_status}
'''

    print(output)

    # Print logs in a file
    if os.environ.get('LOG') == '1':
        with open('api_test.log', 'a') as file:
            file.write(output)
    
# Test pour les utilisateurs existants
test_predict_image('admin', 'boss', 200)




#######################################################################################################

@operations.post("/monitoring_image_class")

async def rep_monitoring_image_class(current_user: str = Depends(authenticate_user)):
    if current_user != "admin":
        raise HTTPException(status_code=403, detail="Seul l'administrateur peut accéder à cette fonctionalité.")
    
    
    # Chemin vers l'image à tester
    img_path = '/home/workspace/API/app/operations/image_monitor.jpg'
    real_code = 40
    
    # Chargement de l'image avec OpenCV en utilisant cv2.IMREAD_COLOR
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Redimensionnement de l'image aux dimensions d'entrée du modèle (299x299)
    img = cv2.resize(img, (299, 299))

    # Modification de la profondeur de l'image en 8 bits (CV_8U) 
    img = img.astype(np.uint8)

    # Normalisation de l'image 
    img = img / 255.0

    # Ajout d'une dimension pour correspondre aux attentes du modèle
    img = np.expand_dims(img, axis=0)

    
    # chargement du modèle pré-entraîné xception
    model = load_model('/home/workspace/API/app/models/checkpoint_Xception_model.h5')
    
    # prédiction avec le modèle xception
    predictionsimage = model.predict(img)
    y_pred_image= predictionsimage * 100
 
    # Liste des labels
    labels = ["Livres adultes", "Jeux Vidéos", "Accessoires de Jeux Vidéos", "Consoles de jeux", "Figurine", "Carte à Collectionner",
            "Masques", "Jouets pour Enfants", "Jeux de Cartes et de société", "Produits télécommandés",
            "Vêtements pour enfants", "Jouets pour Enfants", "Produits Bébés et Enfants",
            "Literies et Meubles", "Accessoires Maison", "Alimentation", "Décoration d'intérieur", "Accessoires Animaux",
            "Journaux et Magazines", "Livres et Revues", "Jeux", "Papeterie",
            "Mobilier", "Piscine", "Jardinage", "Livres", "Jeux en ligne et Logiciels"]
    
    # Code produit associé aux classes
    code_produit = [10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281,
                1300, 1301, 1302, 1320, 1560, 1920, 1940, 2060,
                2220, 2280, 2403, 2462, 2522, 2582, 2583,
                2585, 2705, 2905]
    
    # fonction reshape pour convertir en tableau 1D
    prob_class_img = (y_pred_image.reshape(-1))


    # Création du DataFrame
    data = {'code_produit': code_produit, 'labels': labels, 'target_value_class_image': prob_class_img, 'prob_class_img': prob_class_img}
    data['erreur %'] = (data['target_value_class_image'] - data['prob_class_img']) # Calculez l'erreur en pourcentage
    dfmon = pd.DataFrame(data)

    # Indice de la valeur maximale pour weighted_proba
    idx_max_wgt = dfmon['prob_class_img'].idxmax()
    max_weighted_row = dfmon.loc[idx_max_wgt].to_dict()  
    
    
    # Comparaison entre real_code et la valeur prédite
    predicted_code = int(max_weighted_row['code_produit'])
    is_correct_prediction = real_code == predicted_code
    
    #Vérifiez si l'erreur est supérieure à 5 %
    error = int(max_weighted_row['erreur %'])
    is_error_ok = (error < 10)  # Cela renverra True si au moins une erreur est supérieure à 5 %

    # Ajoutez le résultat de la comparaison dans le dictionnaire
    max_weighted_row['is_correct_prediction'] = is_correct_prediction
    max_weighted_row['is_error_high'] = is_error_ok

    ####################################################################################################
    ######################  test du f1_weighted score sur le dataset ###################################
    f1_target=0.6
    
    #Charger le DataFrame depuis le fichier CSV
    df_test= pd.read_csv('/home/workspace/API/BDD/df_cleaned.csv')
    df_test['text'].fillna("''", inplace=True)
    result = f1_img_test(df_test)
    
    f1_error = int(abs(f1_target - result))
    is_f1_error = (f1_error < 0.2)
    max_weighted_row['is_f1_error'] = is_f1_error


    # Retournez la réponse JSON mise à jour
    return JSONResponse(content=max_weighted_row)


    ######################### ajout du réentrainement"###############################################""
    # Si la prédiction est incorrecte, lancez le réentraînement
    #if not is_correct_prediction:
    #    retrain_img(df_retrain)  # Appelez la fonction de réentraînement

    # Si l'erreur est supérieure à 10% , lancez le réentraînement
    #if not is_error_ok:
    #    retrain_img(df_retrain)  # Appelez la fonction de réentraînement

    # Si le F1 weighted score est inférieur de 10%
    #if not is_f1_error:
        # retrain_img(df_retrain)
        
        
        ############################################################################################
        
    df_retrain = pd.read_csv(df_retrain_path)
    # Si le F1 weighted score est inférieur de 10%
    if not is_f1_error:
        retrain_text(df_retrain)
    
    
    # Fonction pour générer un nom de fichier de journal basé sur la date et l'heure actuelles
    def generate_log_filename():
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'log_{formatted_time}.txt'
        return log_filename

    # Écriture de output_1 et output_2 dans un fichier de journal dans le répertoire spécifié
    log_filename = generate_log_filename()
    log_path = os.path.join(log_directory, log_filename)

    with open(log_path, 'w') as log_file:
        log_file.write(output_1)
        log_file.write(output_2)
        print(f"Log file '{log_path}' created.")
 

