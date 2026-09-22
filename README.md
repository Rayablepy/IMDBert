---
title: IMDBert
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.28.0
app_file: ui.py
pinned: false
license: mit
---

# IMDBert

Fine-tuned **DistilBERT** for binary sentiment classification of IMDB movie reviews (positive / negative).

The base model is trained in `main.ipynb`, saved both locally (`./models/IMDBert`) and uploaded to [Rayhan-08/IMDBert](https://huggingface.co/Rayhan-08/IMDBert) on Hugging Face Hub.

## Features

- **Gradio UI** (`ui.py`) — hosted as a Hugging Face Space via the YAML metadata above.
- **CLI tester** (`tester.py`) — interactive review-by-review predictions in the terminal.
- Auto-detect device (MPS on Mac, CUDA on Windows, else CPU).
- Loads the model from the local folder first, falling back to the Hugging Face Hub automatically.

## Quickstart

```sh
pip install -r requirements.txt
# or: uv sync
```

### Run the Gradio UI

```sh
python ui.py
```

### Run the CLI tester

```sh
python tester.py
```

Type a review and hit enter; press `q` to quit.

### Retrain (optional)

Open `main.ipynb` in Jupyter / Google Colab and run all cells. It downloads the
[IMDB dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews),
trains DistilBERT for 3 epochs, evaluates accuracy/ROC-AUC, and saves the model
locally and to the Hugging Face Hub.

## Project structure

| File                | Purpose                                          |
| ------------------- | ------------------------------------------------ |
| `loader.py`         | Shared model/device loading + prediction helpers |
| `ui.py`             | Gradio app (`app_file` for the HF Space)         |
| `tester.py`         | Terminal-based sentiment tester                  |
| `main.ipynb`        | Training and evaluation notebook                 |
| `requirements.txt`  | Pinned Python dependencies                       |
| `models/IMDBert/`   | Locally saved fine-tuned model (empty until saved) |

## Model

```
Rayhan-08/IMDBert
```

DistilBERT-base-uncased fine-tuned on IMDB with a 2-class classification head.
Inputs are truncated/padded to 256 tokens.