import requests
import base64
import io
import os

def send_csv_file(file_path):
    url = "http://localhost:8080/v1/models/eeg_model:predict"  # Replace with the URL of the predict endpoint

    with open(file_path, 'rb') as file:
        file_content = file.read()

        base64_data = base64.b64encode(file_content).decode('utf-8')

        inference_input = {
            'instances': [{'content': base64_data,
                          'filename': os.path.basename(file_path)}]
        }
        print(inference_input["instances"][0]['filename'])
        response = requests.post(url, json=inference_input)

    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code)
        return None

# Example usage
file_path = "Data_S01_Sess01.csv"  # Replace with the path to your .wav file
prediction_result = send_csv_file(file_path)
if prediction_result is not None:
    print(prediction_result)