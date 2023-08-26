import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

class GroupKFoldmAP:
    def __init__(self, dataset_path, n_splits=4):
        self.dataset_path = dataset_path
        self.n_splits = n_splits
        self.metadata_df = None
        self.bias_df = None
        self.remainder_df = None
        self.selected_columns = ['prompt', 'A', 'B', 'C', 'D', 'E', 'answer', 'mAP']
        self.mAP_classes = {}

    def _discretize_mAP(self, value):
        return self.mAP_classes[value]

    def load_and_preprocess_data(self, bias_threshold=0.5):
        self.metadata_df = pd.read_csv(self.dataset_path)
        self.metadata_df = self.metadata_df.drop_duplicates(subset="prompt", keep='last')
        
        unique_mAP_values = self.metadata_df['mAP'].unique()
        self.mAP_classes = {value: idx for idx, value in enumerate(unique_mAP_values)}

        self.metadata_df['mAP_class'] = self.metadata_df['mAP'].apply(self._discretize_mAP)
        self.bias_df = self.metadata_df[self.metadata_df['mAP'] >= bias_threshold]
        self.remainder_df = self.metadata_df[self.metadata_df['mAP'] < bias_threshold]

    def stratified_group_kfold(self):
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        distributions = []
        
        for fold_number, (_, val_index) in enumerate(sgkf.split(self.bias_df, self.bias_df['mAP_class'], self.bias_df['answer'])):
            val_set = self.bias_df.iloc[val_index][self.selected_columns]
            required_samples_for_train = len(self.metadata_df) - len(val_set)
            
            # Determine training set
            train_sets = [self.remainder_df] if required_samples_for_train > len(self.remainder_df) else []
            remaining_samples_required = required_samples_for_train - len(train_sets[0]) if train_sets else required_samples_for_train
            remaining_bias_df = self.bias_df.drop(self.bias_df.index[val_index])
            train_sets.append(remaining_bias_df.sample(remaining_samples_required, random_state=fold_number))
            
            train_set = pd.concat(train_sets)
            
            # Update fold labels
            train_set['fold'] = fold_number
            val_set['fold'] = fold_number

            distributions.append({
                'fold': fold_number,
                'train_set': train_set,
                'val_set': val_set
            })

        return distributions

    def save_to_csv(self, distributions):
        for dist in distributions:
            fold = dist['fold']
            train_set = dist['train_set']
            val_set = dist['val_set']
            train_set.to_csv(f'train_fold-{fold}.csv', index=False)
            val_set.to_csv(f'val_fold-{fold}.csv', index=False)


if __name__ == "__main__":
    kfold = GroupKFoldmAP('input_file.csv')
    kfold.load_and_preprocess_data(bias_threshold=0.6)  
    distributions = kfold.stratified_group_kfold()
    kfold.save_to_csv(distributions)
