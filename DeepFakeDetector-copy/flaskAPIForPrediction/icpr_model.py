import torch
from PIL import Image
from blazeface import FaceExtractor, BlazeFace
from architectures import fornet, weights
from isplutils import utils

class ICPRModel:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.face_extractor = self.load_face_extractor()
        self.transf = utils.get_transformer('scale', 224, self.model.get_normalizer(), train=False)
    
    def load_model(self):
        net_model = 'Xception'
        net = getattr(fornet, net_model)().eval().to(self.device)
        net.load_state_dict(torch.load("./icpr2020dfdc/Xception_DFDC.pth", map_location=self.device, weights_only=True))
        return net
    
    def load_face_extractor(self):
        facedet = BlazeFace().to(self.device)
        facedet.load_weights("./blazeface/blazeface.pth")
        facedet.load_anchors("./blazeface/anchors.npy")
        return FaceExtractor(facedet=facedet)
    
    def predict(self, image_file):
        image = Image.open(image_file).convert('RGB')
        faces = self.face_extractor.process_image(img=image)
        if not faces['faces']:
            return {"error": "No face found"}
        face = faces['faces'][0]
        face_tensor = self.transf(image=face)['image'].unsqueeze(0)
        with torch.no_grad():
            prediction = torch.sigmoid(self.model(face_tensor.to(self.device))).item()
        return {"prediction": 1-prediction}
