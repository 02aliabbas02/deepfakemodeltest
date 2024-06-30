import torch

def test_model_load():
    model_path = "./Xception_DFDC.pth"
    try:
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

if __name__ == "__main__":
    test_model_load()
