from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import boto3
import json

application = Flask(__name__)
CORS(application)


@application.route('/predict', methods=['POST'])
def predict():
    # Extract JSON data from POST request
    data = request.get_json()

    # Initialize SageMaker runtime client
    runtime = boto3.client('runtime.sagemaker', region_name='us-east-1')

    # Specify SageMaker endpoint
    endpoint_name = 'llama-mental-health-endpoint'

    # Preprocess the payload, so that it isn't too long for the model
    data["text"] = data["text"][:768]

    # Prepare payload for SageMaker
    payload = {"inputs": data["text"]}

    # Invoke SageMaker endpoint
    response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                       ContentType='application/json',
                                       Body=json.dumps(payload))

    # Parse the response from SageMaker
    result = json.loads(response['Body'].read())

    # Extract 'generated_text' from the response
    generated_text = result[0]['generated_text'] if result else "No response generated."

    # Return the 'generated_text'
    return jsonify({"generated_text": generated_text})


@application.route('/health', methods=['GET'])
def health_check():
    # This endpoint simply returns a 200 OK response, indicating the app is healthy.
    return jsonify({"status": "healthy"}), 200


@application.route('/.well-known/pki-validation/<path:filename>')
def serve_well_known_files(filename):
    return send_from_directory('.well-known/pki-validation', filename)


if __name__ == '__main__':
    application.run(debug=True)
