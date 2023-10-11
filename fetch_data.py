from datasets import load_dataset
import logging
# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPipeline:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = None
        self.train_data = None
        self.val_data = None

    def load_data(self):
        """Load dataset using the provided dataset name."""
        self.dataset = load_dataset(self.dataset_name)
        
    def save_train_data(self, path):
        """Save training data to a CSV file."""
        if self.dataset and 'train' in self.dataset:
            self.train_data = self.dataset['train']
            self.train_data.to_csv(path)
        else:
            logging("Training data not available or not loaded.")

    def save_val_data(self, path):
        """Save validation data to a CSV file."""
        if self.dataset and 'validation' in self.dataset:
            self.val_data = self.dataset['validation']
            self.val_data.to_csv(path)
        else:
            logging("Validation data not available or not loaded.")


if __name__ == "__main__":
    pipeline = DataPipeline("natnitaract/kaggel-llm-science-exam-2023-RAG")
    pipeline.load_data()
    pipeline.save_train_data('data/train-val/train.csv')
    pipeline.save_val_data('data/train-val/val.csv')
        