from flask import Flask, request, jsonify
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


if __name__ == '__main__':
    application.run(debug=True)
