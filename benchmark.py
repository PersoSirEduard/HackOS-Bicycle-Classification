from betamark import bicycle
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
from PIL import Image

model_path = "./results/checkpoint-6252"
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def model_infer(x):
    """
    Params:
    -------
    x: NumPy array representation of an image (dimensions are non-fixed)

    Returns:
    --------
    y_pred: int where 0 is negative (no bicycle) or 1 (there is a bicycle)
    """
    image = Image.fromarray(x).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logit = outputs.logits.squeeze().item()
        y_pred = 1 if logit >= 0.5 else 0

    return y_pred

results = bicycle.run_eval(user_func=model_infer)
print(f"Accuracy: {results['acc'] * 100.0}%")
