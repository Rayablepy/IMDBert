from transformers import AutoModelForSequenceClassification, AutoTokenizer
def get_model():
    load_choice=int(input("""
        1.Local download
        2.Huggingface download
        """))
    if load_choice==1:
        try:
            model = AutoModelForSequenceClassification.from_pretrained("./models/IMDBert")
            tokenizer = AutoTokenizer.from_pretrained("./models/IMDBert")
            return model,tokenizer
        except Exception as e:
            print(f"Error occurred while loading model: \n {e}")
            return None
    elif load_choice==2:
        model = AutoModelForSequenceClassification.from_pretrained("Rayhan-08/IMDBert")
        tokenizer = AutoTokenizer.from_pretrained("Rayhan-08/IMDBert")
        return model,tokenizer
    else:
        print("Invalid error, defaulting to HuggingFace, ctrl+C to abort")
        model = AutoModelForSequenceClassification.from_pretrained("Rayhan-08/IMDBert")
        tokenizer = AutoTokenizer.from_pretrained("Rayhan-08/IMDBert")
        return model,tokenizer
