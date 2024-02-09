import warnings
warnings.filterwarnings("ignore")

import logging
import torch
from transformers import AutoModelForMultipleChoice, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from dataclasses import dataclass
from typing import Union, Optional
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

from typing import Optional, Union
import pandas as pd, numpy as np, torch
from datasets import Dataset,load_dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

import config as cfg
import argparse 

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class EvalModel:
    def __init__(self, model_path, max_input=384, name_eval="natnitaract/kaggel-llm-science-exam-2023-RAG"):
        self.model = AutoModelForMultipleChoice.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.MAX_INPUT = max_input
        self.name_eval = name_eval
        self.trainer = Trainer(model=self.model)
        self.option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
        self.index_to_option = {v: k for k, v in self.option_to_index.items()}

    def not_none(self, example):
        fields_to_check = ['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer']
        return all(example[field] is not None for field in fields_to_check)

    def preprocess(self, example):
        context = " ".join(example['context']) if isinstance(example['context'], list) else example['context']
        first_sentence = ["[CLS] " + context] * 5
        prompt = example['prompt'] if isinstance(example['prompt'], str) else " ".join(example['prompt'])
        second_sentences = [
            " #### " + prompt + " [SEP] " + 
            (" ".join(example[option]) if isinstance(example[option], list) else example[option]) + 
            " [SEP]" for option in 'ABCDE'
        ]
        tokenized_example = self.tokenizer(first_sentence, second_sentences, 
                                        padding=True, truncation=True, 
                                        max_length=self.MAX_INPUT, add_special_tokens=False)

        tokenized_example['label'] = self.option_to_index[example['answer']]
        return tokenized_example

    @staticmethod    
    def precision_at_k(r, k):
        """Precision at k"""
        assert k <= len(r)
        assert k != 0
        return sum(int(x) for x in r[:k]) / k
    
    @staticmethod
    def MAP_at_3(predictions, true_items):
        """
        Score is mean average precision at 3
        ref: https://www.kaggle.com/code/philippsinger/h2ogpt-perplexity-ranking
        """
        U = len(predictions)
        map_at_3 = 0.0
        for u in range(U):
            user_preds = predictions[u].split()
            user_true = true_items[u]
            user_results = [1 if item == user_true else 0 for item in user_preds]
            for k in range(min(len(user_preds), 3)):
                map_at_3 += EvalModel.precision_at_k(user_results, k + 1) * user_results[k]
        return map_at_3 / U

    def evaluate(self):
        dataset = load_dataset(self.name_eval)
        val_data = dataset['validation'].filter(self.not_none)

        tokenized_test_dataset = val_data.map(
            self.preprocess,
            remove_columns=val_data.column_names
        )

        predictions = self.trainer.predict(tokenized_test_dataset).predictions
        predictions = np.argmax(predictions, axis=1)

        predictions_as_letters = [self.index_to_option[pred] for pred in predictions]

        true_answers = val_data['answer']
        mAP = self.MAP_at_3(predictions_as_letters, true_answers)
        print(f'CV MAP@3 = {mAP}')
        return mAP

class TrainingPipeline:
    def __init__(self):
        self.option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
        self.index_to_option = {v: k for k, v in self.option_to_index.items()}
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.MAX_INPUT = cfg.MAX_INPUT  
    
    def preprocess(self, example):
        context = " ".join(example['context']) if isinstance(example['context'], list) else example['context']
        first_sentence = ["[CLS] " + context] * 5
        prompt = example['prompt'] if isinstance(example['prompt'], str) else " ".join(example['prompt'])
        second_sentences = [
            " #### " + prompt + " [SEP] " + 
            (" ".join(example[option]) if isinstance(example[option], list) else example[option]) + 
            " [SEP]" for option in 'ABCDE'
        ]
        tokenized_example = self.tokenizer(first_sentence, second_sentences, 
                                        padding=True, truncation=True, 
                                        max_length=self.MAX_INPUT, add_special_tokens=False)

        tokenized_example['label'] = self.option_to_index[example['answer']]
        return tokenized_example

    def not_none(self, example):
        fields_to_check = ['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer']
        return all(example[field] is not None for field in fields_to_check)

    def load_data(self):
        self.dataset = load_dataset("natnitaract/kaggel-llm-science-exam-2023-RAG")
        self.train_data = self.dataset['train'].filter(self.not_none).shuffle(seed=42).select(range(20))
        self.val_data = self.dataset['validation'].filter(self.not_none).shuffle(seed=42).select(range(20))

    def setup_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL, use_fast=True, max_length=512)
        self.model = AutoModelForMultipleChoice.from_pretrained(cfg.MODEL, ignore_mismatched_sizes=True)
        
        if cfg.FREEZE_EMBEDDINGS:
            logging.info('Freezing embeddings.')
            for param in self.model.deberta.embeddings.parameters():
                param.requires_grad = False
        if cfg.FREEZE_LAYERS > 0:
            logging.info(f'Freezing {cfg.FREEZE_LAYERS} layers.')
            for layer in self.model.deberta.encoder.layer[:cfg.FREEZE_LAYERS]:
                for param in layer.parameters():
                    param.requires_grad = False

    def map_dataset(self):
        self.tokenized_train = self.train_data.map(
            self.preprocess,  
            remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'], 
            num_proc = 4
        )
        
        self.tokenized_valid = self.val_data.map(
            self.preprocess, 
            remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'],
            num_proc = 4
        )

    def map_at_3(self, predictions, labels):
        map_sum = 0
        pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
        for x,y in zip(pred,labels):
            z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
            map_sum += np.sum(z)
        return map_sum / len(predictions)

    def compute_metrics(self, p):
        predictions = p.predictions.tolist()
        labels = p.label_ids.tolist()
        return {"map@3": self.map_at_3(predictions, labels)}
        
    def configure_training(self):
        training_args = TrainingArguments(
            warmup_ratio=cfg.WARMUP_RATIO,
            learning_rate=cfg.LR,
            per_device_train_batch_size=cfg.BATCH_SIZE,
            per_device_eval_batch_size=cfg.BATCH_SIZE,
            num_train_epochs=cfg.EPOCHS,
            report_to=cfg.REPORT_TO,
            output_dir = cfg.OUTPUT_DIR,
            overwrite_output_dir=cfg.OVERWRITE_OUTPUT_DIR,
            fp16=cfg.FP16,
            evaluation_strategy=cfg.EVAL_STRATEGY,
            save_strategy=cfg.SAVE_STRATEGY,
            metric_for_best_model=cfg.METRIC_FOR_BEST_MODEL,
            lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
            save_total_limit=cfg.SAVE_TOTAL_LIMIT,
            load_best_model_at_end=cfg.LOAD_BEST_MODEL_AT_END,
            seed=cfg.SEED,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=self.tokenizer),
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_valid,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.EARLY_STOPPING)] 
        )

    def train(self):
        self.trainer.train()

    def save_model(self):
        self.trainer.save_model(cfg.OUTPUT_DIR)

    def evaluate(self, name_eval="natnitaract/kaggel-llm-science-exam-2023-RAG"):
        evaluator = EvalModel(
            model_path=cfg.OUTPUT_DIR, 
            name_eval=name_eval
            )

        return evaluator.evaluate()

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
    
        logging.info("Evaluating model...")
        self.evaluate()
    
        logging.info("Pipeline completed.")
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process")
    parser.add_argument('--train', action='store_true', help="Run training pipeline")
    parser.add_argument('--eval', action='store_true', help="Run evaluation")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    pipeline = TrainingPipeline()

    if args.train:
        logging.info("Running training...")
        pipeline.run_pipeline()
    if args.eval:
        logging.info("Running evaluation...")
        pipeline.evaluate()