import pandas as pd
import logging
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs
import json

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# logging.info("Loading doc data...")
# doc_df = pd.read_csv('./data/News2022_doc_B.tsv', sep='\t', header=None)
# doc_df.columns = ['qid', 'passage']

logging.info("Loading predict data...")
test_df = pd.read_csv('./data/News2022_task2_test.tsv', sep='\t', header=None)
test_df.columns = ['id', 'question']

model_type = "dpr"
context_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"
question_encoder_name = "facebook/dpr-question_encoder-single-nq-base"
train_args = {
    'max_seq_length': 256,
    'train_batch_size': 32,
    'include_title': False
}

model = RetrievalModel(
    model_type='dpr',
    model_name='./output/dpr/checkpoint-2000',
    args=train_args,

)

logging.info("Predicting...")
predicted_passages, doc_ids, doc_vectors, doc_dicts = model.predict(
    test_df['question'].values.tolist(),
    prediction_passages='./output/prediction_passage_dataset',
    retrieve_n_docs=10,
)
test_df['doc_id'] = [json.dumps(list(x)) for x in doc_ids]
test_df.to_csv('result.csv', index=False)