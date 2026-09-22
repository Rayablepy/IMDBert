from loader import get_device, get_model, predict_sentiment

device = get_device()
model, tokenizer = get_model()
model.to(device)
print(f"Model loaded on: {device}")

while True:
    review = input("Review(q to quit): \n")
    if review == "q":
        break
    label, confidence = predict_sentiment(review, model, tokenizer, device)
    print(f"Prediction: {label} ({confidence:.2%} confidence)")
