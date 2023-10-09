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

class EvalModel:
    def __init__(self, model_path, test_file, preprocess_function, max_input=384):
        self.test_file = test_file
        self.preprocess = preprocess_function
        self.MAX_INPUT = max_input
        self.model = AutoModelForMultipleChoice.from_pretrained(model_path)
        self.trainer = Trainer(model=self.model)

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
        test_df = pd.read_csv(self.test_file)
        tokenized_test_dataset = Dataset.from_pandas(test_df).map(
            self.preprocess, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E']
        )
        
        test_predictions = self.trainer.predict(tokenized_test_dataset).predictions
        predictions_as_ids = np.argsort(-test_predictions, 1)
        predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]

        predictions_as_string = test_df['prediction'] = [' '.join(row) for row in predictions_as_answer_letters[:, :3]]
        logging.info(predictions_as_string)

        mAP = EvalModel.MAP_at_3(test_df.prediction.values, test_df.answer.values)
        logging.info(f'CV MAP@3 = {mAP}')
        return mAP

class TrainingPipeline:
    def __init__(self, train_file=cfg.INPUT_TRAIN, val_file=cfg.INPUT_VAL):
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
        self.tokenized_dataset_valid = self.dataset_valid.map(self.preprocess, 
            remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
        self.tokenized_dataset = self.dataset.map(self.preprocess, 
            remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])

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
            evaluation_strategy=cfg.EVALUATION_STRATEGY,
            save_strategy=cfg.SAVE_STRATEGY,
            metric_for_best_model=cfg.METRIC_FOR_BEST_MODEL,
            lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
            save_total_limit=cfg.SAVE_TOTAL_LIMIT,
            seed=cfg.SEED,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=self.tokenizer),
            train_dataset=self.tokenized_dataset,
            eval_dataset=self.tokenized_dataset_valid,
            compute_metrics=ComputeMetrics(),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.EARLY_STOPPING)] 
        )

    def train(self):
        self.trainer.train()

    def save_model(self):
        self.trainer.save_model(cfg.OUTPUT_DIR)

    def evaluate(self, eva_file=cfg.INPUT_EVA):
        evaluator = EvalModel(cfg.OUTPUT_DIR, eva_file, self.preprocess)
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

if __name__ == "__main__":

    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
    pipeline.evaluate()
