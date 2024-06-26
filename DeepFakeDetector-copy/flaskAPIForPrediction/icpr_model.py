import torch
import torchvision.transforms as transforms
from PIL import Image
import os
class ICPRModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
    def load_model(self):
        # Example: Load your PyTorch model
        model_path = './icpr2020dfdc/EfficientNetB4_DFDC.pth'  # Replace with actual path
        model = torch.load(model_path, map_location=self.device)
       # model.eval()  # Set model to evaluation mode
        return model

    def preprocess_image(self, image):
        # Preprocess the image for model input
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image)
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image

    def predict(self, image):
        try:
            # Make prediction with your model
            image_tensor = self.preprocess_image(image).to(self.device)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                # Process outputs as needed
                return outputs.tolist()  # Convert to list for JSON serialization
        except Exception as e:
            return {"error": str(e)}
