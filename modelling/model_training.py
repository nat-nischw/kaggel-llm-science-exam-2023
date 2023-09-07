import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from typing import Optional, Union
import pandas as pd, numpy as np, torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import (AutoTokenizer, EarlyStoppingCallback, 
                          PreTrainedTokenizerBase, PaddingStrategy,
                          AutoModelForMultipleChoice, TrainingArguments, Trainer)

VER = 3
USE_PEFT = False
FREEZE_LAYERS = 18
FREEZE_EMBEDDINGS = True
MAX_INPUT = 1024
MODEL = 'microsoft/deberta-v3-large'

df_valid = pd.read_csv('./60k-data-with-context-v2/train_with_context2.csv')
print('Validation data size:', df_valid.shape)

df_train = pd.read_csv('./60k-data-with-context-v2/all_12_with_context2.csv')
df_train = df_train.drop(columns="source")
df_train = df_train[~df_train['prompt'].isin(df_valid['prompt'])].copy()
print('Train data size:', df_train.shape)

option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k, v in option_to_index.items()}

def preprocess(example):
    first_sentence = ["[CLS] " + example['context']] * 5
    second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first',
                                  max_length=MAX_INPUT, add_special_tokens=False)
    tokenized_example['label'] = option_to_index[example['answer']]
    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

tokenizer = AutoTokenizer.from_pretrained(MODEL)
dataset_valid = Dataset.from_pandas(df_valid)
dataset = Dataset.from_pandas(df_train)
dataset = dataset.remove_columns(["__index_level_0__"])

tokenized_dataset_valid = dataset_valid.map(preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
tokenized_dataset = dataset.map(preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])

model = AutoModelForMultipleChoice.from_pretrained(MODEL, ignore_mismatched_sizes=True)

if FREEZE_EMBEDDINGS:
    print('Freezing embeddings.')
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False
if FREEZE_LAYERS > 0:
    print(f'Freezing {FREEZE_LAYERS} layers.')
    for layer in model.deberta.encoder.layer[:FREEZE_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False

def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1 * np.array(predictions), axis=1)[:, :3]
    for x, y in zip(pred, labels):
        z = [1 / i if y == j else 0 for i, j in zip([1, 2, 3], x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}

training_args = TrainingArguments(
    warmup_ratio=0.8,
    learning_rate=2e-6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    report_to='none',
    output_dir=f'./checkpoints_{VER}',
    overwrite_output_dir=True,
    fp16=True,
    evaluation_strategy='epoch',
    save_strategy="epoch",
    metric_for_best_model='map@3',
    lr_scheduler_type='cosine',
    save_total_limit=2,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_valid,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(f'./model_v{VER}')
