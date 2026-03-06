from model import processor, model
from PIL import Image
import torch
import io

def predict_answer(image_bytes, question):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(image, question, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        answer_id = outputs.logits.argmax(-1).item()

    return model.config.id2label[answer_id]