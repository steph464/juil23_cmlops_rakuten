import os
import requests
import base64
import datetime
import pandas as pd
from f1_model import f1_text_test
from retrain_textmodel import retrain_text

# Définition de l'adresse de l'API et du port
api_address = 'localhost'  # Remplacez par l'adresse de votre API
api_port = 8000

# chemin répertoire du dataset de test
df_test_path= "/home/workspace/API/API_BDD/BDD/dataset/valid_data/df_cleaned.csv"

#chemin du dataset de réentrainement
df_retrain_path="/home/workspace/API/API_BDD/BDD/dataset/valid_data/df_cleaned.csv"

# chemin du répertoire de journal
log_directory = '/home/workspace/API/API_BDD/BDD/retrain_text_model/saved_monitoring_journal_retrain'


def test_predict_text(username: str, password: str, expected_status_code: int):
    auth_header = f'Basic {base64.b64encode(f"{username}:{password}".encode()).decode()}'
    headers = {
        'Authorization': auth_header,
        'accept': 'application/json',
    }

    # Endpoint "/predict_text_class"
    predict_text_url = f'http://{api_address}:{api_port}/Predictions/predict_text_class'
    
    
    # Données de texte de test
    designation: str ='30 Spots Encastrable Orientable Blanc Avec Gu10 Led De 5w Eqv. 40w Blanc Froid 6000k'
    description: str ='<b>Lot de 30 Spots encastrable orientable BLANC avec GU10 LED de 5W &#61; 40W</b> <p><b>Kit contenant: x30 Douilles GU10 &#43; x30 Ampoules GU10 5W &#43; x30 Colerettes Blanche</b></p> <p></p> <p><b>L&#39;Ampoules LED 5W GU10 </b>fait partie d une vaste gamme de lampes d ambiance à LED de longue durée faible consommation et brillamment conçues avec un <b>culot GU10</b> sans avoir à changer votre installation.</p> <p>Avec une remarquable combinaison de technologie haute efficacité et une forme innovante <b>L&#39;Ampoules LED GU10 5W </b>fournit la qualité de lumière que vous attendez.</p> <p>Elle dure jusqu à 25 fois plus longtemps et fournit jusqu à 82 % d économies d énergie.</p> <p>De plus d êtres design <b>l&#39;Ampoules LED GU10 5W </b>à une efficacité de rayonnement pouvant s&#39;étendre jusqu&#39;a 150°. <b>L&#39;Ampoules LED GU10 5W</b> sont les mêmes ou meilleures que les <b>Ampoules E27</b> standards mais elles consomment moins d&#39;énergie.</p> <p>Solution économique et écologique de nouvelle génération utilisation dans les maisons bureaux restaurants hôtels et bâtiments publics.Elle remplace entièrement la lampe incandescence ou halogène standard. </p> <p>Installation facile longue durée de vie.</p> <p></p> <p><b>Avantage:</b></p> <p>- Durée de vie : 25.000 heures<br />- Consomme jusqu à 85 % moins d énergie<br />- Ampoule LED omnidirectionnelle homogène et équilibrée<br />- Allumage instantané</p> <p>L&#39;efficacité énergétique et une longue durée de vie des <b>Ampoules LED GU10</b> engendrent moins de remplacements de lampe en comparaison avec les sources incandescentes et halogènes standards.</p> <p></p> <p></p> <b>Douille en céramique GU10 avec câble gainé 230 Volts - Classe 2</b> <p>Connecteur électrique automatique (pas de vis) avec double connexion pour un montage en parallèle (arrivée pour le spot et départ pour le spot suivant) &#34;Sucre&#34; de protection clipsable Dimensions (longueur x diamètre) &#61; 175 x 30 mm. </p> <p>Conforme aux Normes CE - RoHS - EMC <br /><br />Douille pour ampoule à culot GU10/GZ10 automatique<br />Corps isolant en steatite<br />Fil silicone 075mm²<br />Câble gainé<br />Sucre de protection avec fermeture à clipser ( pas besoin de vis ) amovible.<br />Double connexion pour montage en série clipsable sans besoin de s&#39;aider d&#39;un outil.<br />Longueur 15cm <br />Résistant à 250°C</p> <p></p> <p></p> <b>Collerette Orientable Ronde Blanche </b> <p>Dimensions : diamètre 65mm Perçage d&#39;encastrement : pour 90mm de diamètre total </p> <p>Ce support encastrable blanc est idéale grâce à son centre orientable. Cela permet de diffuser la lumière dans une direction choisie. Son diamètre de 90 mm et le trou de coupe de 65 mm. L&#39;ampoule se place par l&#39;avant très pratique lorsque l&#39;on veut la changer. </p> <p>Sa couleur blanche permet de l&#39;intégrer sans problème à votre plafond blanc ou coloré.</p> <p></p>  <p>Puissance</p>  <p>5W</p>  <p>Douille</p>  <p>GU10</p>  <p>Alimentation</p>  <p>AC175-265V</p>  <p>Flux lumineux</p>  <p>400 Lm</p>  <p>Puissance des LEDs</p>  <p>5W</p>  <p>Fréquence de fonctionnement</p>  <p>50-60Hz</p>  <p>Couleur de la lumière</p>  <p>4500K</p>  <p>Angle de la lumière</p>  <p>110°</p>  <p>Ra</p>  <p>&gt;80</p>  <p>Consommation</p>  <p>5KWh/1000h</p>  <p>Température de fonctionnement</p>  <p>-30°C / &#43;50°C</p>  <p>Matériel</p>  <p>Thermoplastic &#43; Aluminium</p>  <p>Degré de protection</p>  <p>IP20</p>  <p>Taille</p>  <p>?50x57 mm</p>  <p>Poids</p>  <p>0.077 kg</p>  <p>Exploitation</p>  <p>&gt; 25 000 Heures</p>'   
    real_code= 2060
    
    # Paramètres de la requête
    params = {
        'designation': designation,
        'description': description,
    }

    response = requests.post(predict_text_url, headers=headers, params=params)
    
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
    
    output_1 = f'''
====================================================================
    Predictions test using CONV1D BASED MODEL - results
====================================================================

request done at "predict_text_class"
| username = '{username}'
| password = '{password}'
| designation = '{designation}'
| description = '{ description}'

expected result code = {expected_status_code}
expected Predicted class code =  {real_code}

actual result code = {status_code}
actual Predicted class code = {predicted_code}  
==> Code result = {test_status}
==> Prediction result = {prediction_status}
'''

    ####################################################################################################
    ######################  test du f1_weighted score sur le dataset ###################################
    f1_target= 0.8
    #Charger le DataFrame depuis le fichier CSV
    df_test= pd.read_csv(df_test_path)
    df_test['text'].fillna("''", inplace=True)
    result = f1_text_test(df_test)
    
    f1_error = abs(f1_target- result)
    is_f1_error = (f1_error < 0.15)
    
    output_2 = f'''
====================================================================
    f1_weighted score monitoring - results
====================================================================

request done at "predict_text_class"
| username = '{username}'
| password = '{password}'

f1_weighted score ={ result}
f1_target = {f1_target}
f1_error (must be < 0.15)= {f1_error}
==> model is valid ={is_f1_error}
'''
   
    # Affichage des résultats
    print(output_1)
    print(output_2)
    
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
 
    
# Test pour les utilisateurs existants
test_predict_text('admin', 'boss', 200)

