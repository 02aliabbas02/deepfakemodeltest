# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.model_zoo import load_url
from PIL import Image
import sys
from blazeface import FaceExtractor, BlazeFace
from architectures import fornet, weights
from isplutils import utils

def main():
    # Initialization
    net_model = 'Xception'
    train_db = 'DFDC'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    face_policy = 'scale'
    face_size = 224
    
     # Model Loading
    # model_url = weights.weight_url['{:s}_{:s}'.format(net_model, train_db)]
    net = getattr(fornet, net_model)().eval().to(device)
    # net.load_state_dict(load_url(model_url, map_location=device, check_hash=True))
    net.load_state_dict(torch.load("/Users/akhtarabbas/Downloads/models/DeepFakeDetector-copy/flaskAPIForPrediction/icpr2020dfdc/Xception_DFDC.pth", map_location=device))
  
    # Load Images
    im_real = Image.open('test_images/real/real00240.jpg')
    im_fake = Image.open('test_images/df/fake.png').convert('RGB')

    # Face Extraction
    facedet = BlazeFace().to(device)
    facedet.load_weights("./blazeface/blazeface.pth")
    facedet.load_anchors("./blazeface/anchors.npy")
    face_extractor = FaceExtractor(facedet=facedet)

    im_real_faces = face_extractor.process_image(img=im_real)
    im_fake_faces = face_extractor.process_image(img=im_fake)
    im_real_face = im_real_faces['faces'][0] if im_real_faces['faces'] else None
    im_fake_face = im_fake_faces['faces'][0] if im_fake_faces['faces'] else None

    if im_real_face is None or im_fake_face is None:
        print("No faces found in one or both images.")
        return
    
    # Transformation and Prediction
    transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)
    faces_t = torch.stack([transf(image=im)['image'] for im in [im_real_face, im_fake_face]])

    with torch.no_grad():
        faces_pred = torch.sigmoid(net(faces_t.to(device))).cpu().numpy().flatten()

    # Print Scores
    print('Score for REAL face: {:.4f}'.format(faces_pred[0]))
    print('Score for FAKE face: {:.4f}'.format(faces_pred[1]))

if __name__ == "__main__":
    main()
