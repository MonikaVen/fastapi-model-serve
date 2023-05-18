from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import pickle5 as pickle
import torch
import clip

#models are saved in either joblib or pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)

class_names = []
with open('classes.txt', 'r') as f: 
    for line in f:
        class_name = line.strip()
        class_names.append(class_name)

# Define the FastAPI app
app = FastAPI()

def get_classifier(file_name: str):
    with open(file_name, 'rb') as file:
        classifier_ = pickle.load(file)
    return classifier_

file_name = 'clip_food101_model.pkl'
classifier = get_classifier(file_name)

def get_features(image):
    all_features = []
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_input.to(device))
        all_features.append(features)
    print(all_features)
    return torch.cat(all_features).cpu().numpy()

# Define the prediction function
def predict_image(image: Image.Image) -> int:
    # Preprocess the image
    features = get_features(image)
    # Make predictions using the loaded model
    prediction = classifier.predict(features)
    prob = classifier.predict_proba(features)
    print(prob[0])
    return int(prediction[0]), prob[0][int(prediction)]

# Define the prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and decode the uploaded image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    # Validate the file format
    allowed_formats = ('.jpeg', '.jpg', '.png')
    if file.filename.lower().endswith(allowed_formats):
        # Make predictions
        prediction, prob = predict_image(image)
        #macarons - for everything not food
        return {"prediction": class_names[prediction], "confidence": prob}
    else:
        return {"error": "Invalid file format. Only JPEG, JPG, and PNG images are supported."}
