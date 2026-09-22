from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_choice=int(input("""
    1.Local download
    2.Huggingface download
    """))
if load_choice==1:
    model = AutoModelForSequenceClassification.from_pretrained("./models/IMDBert")
    tokenizer = AutoTokenizer.from_pretrained("./models/IMDBert")
elif load_choice==2:
    model = AutoModelForSequenceClassification.from_pretrained("Rayhan-08/IMDBert")
    tokenizer = AutoTokenizer.from_pretrained("Rayhan-08/IMDBert")
else:
    print("Invalid option")
