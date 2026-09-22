import gradio as gr
from loader import classify, get_device, get_model

DEVICE = get_device()
MODEL, TOKENIZER = get_model()
MODEL.to(DEVICE)
print(f"Model loaded on: {DEVICE}")


def classify_wrapper(review):
    return classify(review, MODEL, TOKENIZER, DEVICE)


demo = gr.Interface(
    fn=classify_wrapper,
    inputs=gr.Textbox(lines=4, placeholder="Type a movie review here..."),
    outputs=[
        gr.Label(num_top_classes=2),
        gr.Textbox(label="Confidence"),
    ],
    title="IMDBert Sentiment Analysis",
    description="DistilBERT fine-tuned on IMDB movie reviews to predict Positive/Negative sentiment.",
)

if __name__ == "__main__":
    demo.launch()