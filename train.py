import logging
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from dataclasses import dataclass
from typing import Union, Optional
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

from typing import Optional, Union
import pandas as pd, numpy as np, torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

import config as cfg


# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ComputeMetrics:
    def __init__(self, preds, labels):
        self.preds = preds
        self.labels = labels

    @staticmethod
    def map_at_3(predictions, labels):
        map_sum = 0
        pred = np.argsort(-1 * np.array(predictions), axis=1)[:, :3]
        for x, y in zip(pred, labels):
            z = [1/i if y==j else 0 for i, j in zip([1, 2, 3], x)]
            map_sum += np.sum(z)
        return map_sum / len(predictions)

    def compute(self, p):
        predictions = p.predictions.tolist()
        labels = p.label_ids.tolist()
        return {"map@3": self.map_at_3(predictions, labels)}

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

class TrainingPipeline:
    def __init__(self, train_file, val_file):
        self.train_file = train_file
        self.val_file = val_file
        self.option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
        self.index_to_option = {v: k for k, v in self.option_to_index.items()}

    def preprocess(self, example):
        """ Preprocess a single example"""
        first_sentence = [ "[CLS] " + example['context'] ] * 5
        second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
        tokenized_example = self.tokenizer(first_sentence, second_sentences, truncation='only_first',
                                    max_length=cfg.MAX_INPUT, add_special_tokens=False)
        tokenized_example['label'] = self.index_to_optionx[example['answer']]

        return tokenized_example

    def load_data(self):
        self.dataset_valid = Dataset.from_pandas(self.val_file)
        self.dataset = Dataset.from_pandas(self.train_file)
        self.dataset = self.dataset.remove_columns(["__index_level_0__"])

    def setup_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL, use_fast=True, max_length=512)
        self.model = AutoModelForMultipleChoice.from_pretrained(cfg.MODEL, ignore_mismatched_sizes=True)
        
        if cfg,FREEZE_EMBEDDINGS:
            logging.info('Freezing embeddings.')
            for param in self.model.deberta.embeddings.parameters():
                param.requires_grad = False
        if cfg,FREEZE_LAYERS > 0:
            logging.info(f'Freezing {cfg.FREEZE_LAYERS} layers.')
            for layer in self.model.deberta.encoder.layer[:cfg.FREEZE_LAYERS]:
                for param in layer.parameters():
                    param.requires_grad = False

    def map_dataset(self):
        self.tokenized_dataset_valid = self.dataset_valid.map(self.preprocess, 
            remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
        self.tokenized_dataset = self.dataset.map(self.preprocess, 
            remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])

    def configure_training(self):
        training_args = TrainingArguments(
            warmup_ratio=0.8,
            learning_rate=2e-6,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=3,
            num_train_epochs=2,
            report_to='none',
            output_dir = f'output/checkpoints_{cfg.VER}',
            overwrite_output_dir=True,
            fp16=True,
            evaluation_strategy='epoch',
            save_strategy="epoch",
            metric_for_best_model='map@3',
            lr_scheduler_type='cosine',
            save_total_limit=2,
            seed=42,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=self.tokenizer),
            train_dataset=self.tokenized_dataset,
            eval_dataset=self.tokenized_dataset_valid,
            compute_metrics=ComputeMetrics(),
        )

    def train(self):
        self.trainer.train()

    def save_model(self):
        self.trainer.save_model(cfg.OUTPUT_DIR)

    def run_pipeline(self):
        logging.info("Setting up model and tokenizer...")
        self.setup_model_and_tokenizer()
    
        logging.info("Loading data...")
        self.load_data()
    
        logging.info("Mapping dataset...")
        self.map_dataset()
    
        logging.info("Configuring training...")
        self.configure_training()
    
        logging.info("Starting training...")
        self.train()

        logging.info("Saving model...")
        self.save_model()
    
        logging.info("Pipeline completed.")

if __name__ == "__main__":

    
    train_file = "./data/train-stem-wiki.csv"
    val_file = "./data/val-stem-wiki.csv"

    pipeline = TrainingPipeline(train_file, val_file)
    pipeline.run_pipeline()
