from asyncio.log import logger
from turtle import title
import pandas as pd
import csv
import logging
import numpy as np
import os
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

logging.info("Loading doc data...")
doc_df = pd.read_csv('./data/News2022_doc_B.tsv', sep='\t', header=None)
doc_df.columns = ['qid', 'passage']

dataset = {}
for _type in ['train', 'dev']:
    if not os.path.exists(f'./data/{_type}.dpr.cache'):
        logging.info(f"Loading {_type} data...")
        df = pd.read_csv(f'./data/News2022_task2_{_type}.tsv', sep='\t', header=None, quoting=csv.QUOTE_NONE)

        df.columns = ['index', 'text_a', 'labels', 'qid', 'text_b']
        df = df.drop(columns=['index', 'labels', 'text_b'])
        data = []
        for vo in df.to_dict('records'):
            try:
                data.append(dict(
                    query_text = vo['text_a'],
                    gold_passage = doc_df[doc_df['qid'] == int(vo['qid'])]['passage'].values[0]
                ))
            except:
                continue
        with open(f'./data/{_type}.dpr.cache', 'wb') as f:
            np.save(f, data)
    else:
        logger.info("Loading cached data...")
        with open(f'data/{_type}.dpr.cache', 'rb') as f:
            data = list(np.load(f, allow_pickle=True))
    logger.info(f"demo {_type} data: {str(data[-1])}")
    logger.info(f"{_type} data size: {len(data)}")
    dataset[_type] = pd.DataFrame(data)

model_type = "dpr"
context_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"
question_encoder_name = "facebook/dpr-question_encoder-single-nq-base"
train_args = {
    'evaluate_during_training': False,
    'evaluate_during_training_verbose': False,
    'max_seq_length': 256,
    'num_train_epochs': 10,
    'train_batch_size': 32,
    'use_multiprocessing': False,
    'use_multiprocessing_for_evaluation': False,
    'overwrite_output_dir': True,
    'evaluate_during_training_steps': 500,
    'include_title': False
}
model = RetrievalModel(
    model_type='dpr',
    # context_encoder_name='./output/roformer/',
    # query_encoder_name='./output/roformer/',
    args=train_args,
    context_encoder_name=context_encoder_name,
    query_encoder_name=question_encoder_name,
)

logger.info("Training model...")
model.train_model(dataset['train'], output_dir = './output/dpr', 
                  additional_eval_passages=doc_df['passage'].values.tolist())