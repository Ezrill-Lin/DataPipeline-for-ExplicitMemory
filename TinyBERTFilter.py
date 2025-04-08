import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset, Dataset, load_from_disk
from tqdm import tqdm
import json

class TinyBERTFilter():
    def __init__(self, model_path, data_to_label, trial=False):
        self.model_path = model_path
        self.data_to_label = data_to_label
        self.trial = trial
        self.tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labelled_data = []
    
    def start_label(self):
        if self.trial:
            input_texts = []
            with open(self.data_to_label, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 100:
                        break
                    input_texts.append(json.loads(line))
        else:
            with open(self.data_to_label, 'r', encoding='utf-8') as f:
                input_texts = [json.loads(line) for line in f]

        self.model.to(self.device)
        self.model.eval()
        for row in tqdm(input_texts, desc="Labelling"):
            text = row['text']
            index = row['index']
            dataset = row['dataset']
            inputs = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**inputs)
                label = output.logits.argmax(axis=-1).item()
                self.labelled_data.append({
                    'index': index,
                    'label': label,
                    'dataset': dataset,
                    'text': text,
                })
    
    def save_labelled_data(self):
        filtered_data = []
        for line in tqdm(self.labelled_data, desc='Wrapping'):
            if line['label'] == 1:
                filtered_data.append(line)

        filtered_data = Dataset.from_list(filtered_data)
        filtered_data.save_to_disk('labelled_data_(filtered)')



if __name__ == '__main__':
    fitler_trial = TinyBERTFilter(model_path='./tinybert-filter', 
                                  data_to_label='./jsonl/all_data_to_bert.jsonl',
                                  trial=True)
    fitler_trial.start_label()
    print(Dataset.from_list(fitler_trial.labelled_data))
