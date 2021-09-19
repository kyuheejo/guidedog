from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import dotenv_values

config = dotenv_values()

def image_captioning(img):
    subscription_key = config["subscription_key"]
    endpoint = config["endpoint"]

    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    print("===== Describe an image - remote =====")
    # Call API
    description_results = computervision_client.describe_image_in_stream(img)
    print(description_results)
    # Get the captions (descriptions) from the response, with confidence level
    print("Description of remote image: ")
    if len(description_results.captions) == 0:
        return "No description detected."
    return description_results.captions[0].text
