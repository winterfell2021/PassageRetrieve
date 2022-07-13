from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import json

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

logging.info("Loading doc data...")
doc_df = pd.read_csv('./data/News2022_doc_B.tsv', sep='\t', header=None)
doc_df.columns = ['qid', 'passage']

logging.info("build test data...")
test_df = pd.read_csv('./result.csv')
test_ori_df = pd.read_csv('./data/News2022_task2_test.tsv', sep='\t', header=None)
test_ori_df.columns = ['id', 'question']
test_list = []
clue_list = []
for vo in test_df.to_dict('records'):
    doc_id = json.loads(vo['doc_id'])
    doc_id = [int(x) for x in doc_id]
    while doc_df[doc_df['qid'] == doc_id[0]].empty:
        doc_id.pop(0)
        print('pop doc_id')
    test_list.append(
        vo['question'] + ' ' + doc_df[doc_df['qid'] == doc_id[0]]['passage'].values[0]
    )
    clue_list.append(
        doc_df[doc_df['qid'] == doc_id[0]]['passage'].values[0]
    )

logging.info("test sample: {}".format(test_list[-1]))
# Set training arguments
train_args = {
    'max_seq_length': 256,
    'labels_list': [0, 1],
}

# Create model
model = ClassificationModel('auto', './output/classification/final/', num_labels=2, args=train_args, use_cuda=True)
preds, model_outputs = model.predict(test_list)
print(preds)


df = pd.DataFrame()
df['question'] = test_ori_df['question']
df['label'] = preds
df['clue'] = clue_list
df['label'] = df['label'].apply(lambda x: 'Yes' if x else 'No')
df.reset_index(inplace=True)
df.to_csv('submit.tsv', index=False,sep='\t', header=None)