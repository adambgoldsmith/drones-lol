from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import base64
import io
from PIL import Image
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class SketchRequest(BaseModel):
    image_data: str  # Base64 encoded image

# Load model
class SketchRecognitionModel(nn.Module):
    def __init__(self):
        super(SketchRecognitionModel, self).__init__()
        # Define your model architecture here
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 100)  # Adjust number of classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize model
model = SketchRecognitionModel()
model.load_state_dict(torch.load('pytorch_model.bin', map_location=torch.device('cpu')))
model.eval()

# Load class labels (adjust path as needed)
with open('class_labels.txt', 'r') as f:
    class_labels = [line.strip() for line in f]

def preprocess_image(image_data):
    """Convert base64 image to tensor."""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Resize to expected input size
        image = image.resize((64, 64))
        
        # Convert to tensor and normalize
        tensor = torch.tensor(np.array(image), dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0) / 255.0
        
        return tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@app.post("/predict")
async def predict_sketch(request: SketchRequest):
    try:
        # Preprocess image
        input_tensor = preprocess_image(request.image_data)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            predictions = [
                {
                    "label": class_labels[idx],
                    "probability": float(prob)
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
            
        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)