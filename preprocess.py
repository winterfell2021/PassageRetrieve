from os.path import join
from tqdm import tqdm
if __name__ == '__main__':
    maxlen = 512
    data_dir = './data'
    data_list = []

    for name in ["train", "dev", "test"]:
        with open(join(data_dir, "News2022_{}.tsv".format(name)), "r", encoding="utf8") as fr:
            for line in tqdm(fr):
                if 'query' in line:
                    continue
                data_list.append(
                    " ".join(line.strip().split("\t")[-1].strip().split(" ")[:int(maxlen * 1.1)]))

    print(data_list[0])
    with open("data/News2022_doc.tsv", "r", encoding="utf8") as fr:
        for line in tqdm(fr):
            data_list.append(" ".join(line.strip().split(
                "\t")[-1].strip().split(" ")[:int(maxlen * 1.1)]))
    with open(join(data_dir, "full.txt"), 'w') as f:
        f.write('\n'.join(data_list))
