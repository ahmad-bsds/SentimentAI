# import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", model="FacebookAI/roberta-large-mnli", device=0)
# dataset = datasets.load_dataset("superb", name="asr", split="test")

pipe(["This restaurant is awesome", "This restaurant is awful"])
