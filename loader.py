import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LOCAL_PATH = "./models/IMDBert"
HUB_ID = "Rayhan-08/IMDBert"
MAX_LENGTH = 256


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_local():
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
    return model, tokenizer


def load_hub():
    model = AutoModelForSequenceClassification.from_pretrained(HUB_ID)
    tokenizer = AutoTokenizer.from_pretrained(HUB_ID)
    return model, tokenizer


def get_model(source="auto"):
    if source == "local":
        return load_local()
    if source == "huggingface":
        return load_hub()
    if source != "auto":
        raise ValueError(f"Unknown source: {source}")
    try:
        return load_local()
    except Exception as e:
        print(f"Local model not found ({e}). Falling back to HuggingFace.")
        return load_hub()


def predict_sentiment(text, model, tokenizer, device):
    model.eval()

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "positive" if pred == 1 else "negative"
    return label, confidence


def classify(review, model, tokenizer, device):
    if not review.strip():
        return "", ""
    label, confidence = predict_sentiment(review, model, tokenizer, device)
    return label, f"{confidence:.2%}"
