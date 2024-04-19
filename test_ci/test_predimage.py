import os
import requests
import base64

# Définition de l'adresse de l'API et du port
api_address = 'localhost'  
api_port = 8005

def test_predict_image(username: str, password: str, expected_status_code: int):
    auth_header = f'Basic {base64.b64encode(f"{username}:{password}".encode()).decode()}'
    headers = {
        'Authorization': auth_header,
        'accept': 'application/json',
    }

    # Endpoint "/predict_image_class"
    predict_image_url = f'http://{api_address}:{api_port}/Predictions/predict_image_class'

    # Chemin vers l'image à tester
    image_path = '../test_ci/image_test/image_938157330_product_199450457.jpg'
    real_label = "jeux vidéos"
    real_code = 40

    # Vérifier si le fichier image existe
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return

    # Envoi de la requête POST avec l'image
    with open(image_path, 'rb') as image_file:
        files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}

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
