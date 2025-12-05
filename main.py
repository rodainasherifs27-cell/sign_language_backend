
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json
import uvicorn
from typing import List

app = FastAPI(title="Sign Language AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing Sign Language AI Model...")

try:
    model = tf.keras.models.load_model('model/sign_language_model_final.h5')
    with open('model/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    with open('model/model_info.json', 'r') as f:
        model_info = json.load(f)
    print("Model loaded successfully")
    print(f"Model configured for {len(class_mapping)} sign language classes")
    print(f"Classes: {list(class_mapping.keys())}")

except Exception as e:
    print(f"Error loading model: {e}")
    raise e

def preprocess_data(input_data: List[float]) -> np.ndarray:
    data_array = np.array(input_data, dtype=np.float32)
    
    if len(data_array.shape) > 1:
        data_array = data_array.flatten()
    
    target_size = model_info['feature_size']
    if len(data_array) > target_size:
        data_array = data_array[:target_size]
    elif len(data_array) < target_size:
        padded = np.zeros(target_size, dtype=np.float32)
        padded[:len(data_array)] = data_array
        data_array = padded
    
    data_array = (data_array - np.mean(data_array)) / (np.std(data_array) + 1e-8)
    
    return data_array.reshape(1, -1)

@app.get("/")
async def root():
    return {
        "message": "Sign Language AI API is running",
        "classes": list(class_mapping.keys()),
        "status": "active"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict(data: List[float]):
    try:
        if len(data) == 0:
            raise HTTPException(status_code=400, detail="No data provided")
        
        processed_data = preprocess_data(data)
        
        predictions = model.predict(processed_data)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        class_name = list(class_mapping.keys())[predicted_class_idx]
        
        return {
            "success": True,
            "predicted_class": predicted_class_idx,
            "class_name": class_name,
            "confidence": confidence,
            "all_predictions": predictions[0].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    return {
        "input_shape": model_info['input_shape'],
        "num_classes": model_info['num_classes'],
        "classes": model_info['classes'],
        "feature_size": model_info['feature_size']
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
