from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.metrics import classification_report
import logging
import csv

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Load training data
train_df = pd.read_csv('./data/News2022_task2_train.tsv', sep='\t', header=None, quoting=csv.QUOTE_NONE)
train_df.columns = ['index', 'text_a', 'labels', 'qid', 'text_b']
train_df = train_df.drop(columns=['index', 'qid'])
train_df['labels'] = train_df['labels'].apply(lambda x: 1 if x=='Yes' else 0)
dev_df = pd.read_csv('./data/News2022_task2_dev.tsv', sep='\t', header=None, quoting=csv.QUOTE_NONE)
dev_df.columns = ['index', 'text_a', 'labels', 'qid', 'text_b']
dev_df = dev_df.drop(columns=['index', 'qid'])
dev_df['labels'] = dev_df['labels'].apply(lambda x: 1 if x=='Yes' else 0)


# Set training arguments
train_args = {
    'evaluate_during_training': True,
    'evaluate_during_training_verbose': True,
    'max_seq_length': 256,
    'num_train_epochs': 10,
    'train_batch_size': 128,
    'labels_list': [0, 1],
    'use_multiprocessing': False,
    'use_multiprocessing_for_evaluation': False,
    'overwrite_output_dir': True,
    'evaluate_during_training_steps': 315,
    'sliding_window': False
}

# Create model
model = ClassificationModel('auto', './output/roformer/', num_labels=2, args=train_args, use_cuda=True)

# Define metric
def clf_report(labels, preds):
    return classification_report(labels, preds, output_dict=True)


# Train model 
# Checkpoint after each epoch will be saved to outputs/
# The best model on dev set will be saved to outputs/best_model/
model.train_model(train_df, eval_df=dev_df, clf_report=clf_report, output_dir="output/classification")