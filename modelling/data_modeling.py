import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    def visualize_distributions(self, distributions):
        # Extracting data
        folds = [item['fold'] for item in distributions]

        pos_train = [item['train_set']['mAP'].sum() for item in distributions]
        neg_train = [len(item['train_set']) - item['train_set']['mAP'].sum() for item in distributions]
        pos_val = [item['val_set']['mAP'].sum() for item in distributions]
        neg_val = [len(item['val_set']) - item['val_set']['mAP'].sum() for item in distributions]


        # Plotting
        bar_width = 0.35
        index = range(len(folds))

        fig, ax = plt.subplots(figsize=(10, 6))
        bar1 = ax.bar(index, pos_train, bar_width, label='Train Positive', color='b')
        bar2 = ax.bar(index, neg_train, bar_width, bottom=pos_train, label='Train Negative', color='r')
        bar3 = ax.bar([i + bar_width for i in index], pos_val, bar_width, label='Val Positive', color='c')
        bar4 = ax.bar([i + bar_width for i in index], neg_val, bar_width, bottom=pos_val, label='Val Negative', color='m')

        def label_percentages(bar1, bar2, ax):
            """Attach a percentage label for each segment of the stacked bars."""
            for i in range(len(bar1)):
                pos_height = bar1[i].get_height()
                neg_height = bar2[i].get_height()
                total = pos_height + neg_height
                pos_percentage = (pos_height / total) * 100
                neg_percentage = (neg_height / total) * 100

                ax.annotate(f'{pos_percentage:.1f}%',
                            xy=(bar1[i].get_x() + bar1[i].get_width() / 2, pos_height / 2),
                            ha='center', va='center')
                ax.annotate(f'{neg_percentage:.1f}%',
                            xy=(bar2[i].get_x() + bar2[i].get_width() / 2, pos_height + neg_height / 2),
                            ha='center', va='center')

        label_percentages(bar1, bar2, ax)
        label_percentages(bar3, bar4, ax)

        ax.set_xlabel('Fold Number')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Distribution of samples across folds (in %)')
        ax.set_xticks([i + bar_width/2 for i in index])
        ax.set_xticklabels([f'Fold {i}' for i in folds])
        ax.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    kfold = GroupKFoldmAP('input_file.csv')
    kfold.load_and_preprocess_data(bias_threshold=0.5)  
    distributions = kfold.stratified_group_kfold()
    kfold.visualize_distributions(distributions)
    kfold.save_to_csv(distributions)
