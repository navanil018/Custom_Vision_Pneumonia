from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

ENDPOINT = "https://detectingpneumoniainchestxray.cognitiveservices.azure.com/"

# Replace with a valid key
#training_key = "<your training key>"
prediction_key = "4ad249d8624d4cfcae3c53cae1ce364c"
#prediction_resource_id = "<your prediction resource id>"

publish_iteration_name = "Iteration2"
# Now there is a trained endpoint that can be used to make a prediction
predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

with open("C:/Users/Navanil/Desktop/Kaggle/Pneumonia/test.jpeg", "rb") as image_contents:
    results = predictor.classify_image("66d04970-1bea-4686-b04f-a6c6c1e384f2", publish_iteration_name, image_contents.read())

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
              ": {0:.2f}%".format(prediction.probability * 100))