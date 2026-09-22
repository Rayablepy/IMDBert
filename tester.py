from loader import get_model
import torch.nn.functional as F
import torch
def get_device():
    devtype=input("M for mac, W for windows: ").upper()
    if devtype == "M":
      device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif devtype=="W":
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      device="cpu"
    print(f"Using device: {device}")
    return device
def predict_sentiment(text, model, tokenizer, device):
    model.eval()

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "positive" if pred == 1 else "negative"
    return label, confidence

device=get_device()
model,tokenizer=get_model()
if model is None or tokenizer is None:
    raise SystemExit("Failed to load model. Try Huggingface download (option 2).")
model.to(device)
while True:
    review = input("Review(q to quit): \n")
    if review=="q":
        break
    label, confidence = predict_sentiment(review, model, tokenizer, device)
    print(f"Prediction: {label} ({confidence:.2%} confidence)")
