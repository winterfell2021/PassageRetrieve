from simpletransformers.classification import ClassificationModel
import pandas as pd
import numpy as np 
from sklearn.metrics import classification_report
import logging
import csv
import os

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

if not os.path.exists(f'./data/train.cls.cache') or not os.path.exists(f'./data/dev.cls.cache'): 
    logging.info("Loading doc data...")
    doc_df = pd.read_csv('./data/News2022_doc_B.tsv', sep='\t', header=None)
    doc_df.columns = ['qid', 'passage']


dataset = []
for _type in ['train', 'dev']:
    if not os.path.exists(f'./data/{_type}.cls.cache'):
        logging.info(f"Loading {_type} data...")
        df = pd.read_csv(f'./data/News2022_task2_{_type}.tsv', sep='\t', header=None, quoting=csv.QUOTE_NONE)

        df.columns = ['index', 'text_a', 'labels', 'qid', 'text_b']
        df = df.drop(columns=['index', 'text_b'])
        df['labels'] = df['labels'].apply(lambda x: 1 if x=='Yes' else 0)
        data = []
        for vo in df.to_dict('records'):
            try:
                data.append(
                    dict(
                        text_a=vo['text_a'] + " " + doc_df[doc_df['qid'] == int(vo['qid'])]['passage'].values[0],
                        label=vo['labels']
                    ))
            except:
                continue
        with open(f'./data/{_type}.cls.cache', 'wb') as f:
            np.save(f, data)
    else:
        logging.info("Loading cached data...")
        with open(f'data/{_type}.cls.cache', 'rb') as f:
            data = list(np.load(f, allow_pickle=True))
    logging.info(f"demo {_type} data: {str(data[-1])}")
    logging.info(f"{_type} data size: {len(data)}")
    dataset.extend(data)

train_df = pd.DataFrame(dataset)
logging.info(f"train_df size: {len(train_df)}")
logging.info("Training model...")
# Set training arguments
train_args = {
    'evaluate_during_training': False,
    'max_seq_length': 256,
    'num_train_epochs': 10,
    'train_batch_size': 64,
    'labels_list': [0, 1],
    'use_multiprocessing': False,
    'use_multiprocessing_for_evaluation': False,
    'sliding_window': False,
    'overwrite_output_dir': True,
    'wandb_project': 'DFPassageRetrieve'
}

# Create model
model = ClassificationModel('auto', './output/roformer/', num_labels=2, args=train_args, use_cuda=True)

# Define metric
def clf_report(labels, preds):
    return classification_report(labels, preds, output_dict=True)


# Train model 
# Checkpoint after each epoch will be saved to outputs/
# The best model on dev set will be saved to outputs/best_model/
model.train_model(train_df, clf_report=clf_report, output_dir="output/classification")