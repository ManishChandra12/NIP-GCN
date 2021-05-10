from nltk.corpus import stopwords
import nltk
import sys
from src.utils.utils import clean_str


if len(sys.argv) != 2:
    sys.exit("Use: python -m src.remove_words <dataset>")

datasets = ['dblp', 'M10', 'covid', 'covid_title']
dataset = sys.argv[1]

if dataset not in datasets:
    sys.exit("wrong dataset name")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

doc_content_list = []
doc_id_list = []
with open('data/' + dataset + '/docs.txt', 'rb') as f:
    for line in f.readlines():
        temp = line.strip().decode('latin1').split(' ', 1)
        doc_content_list.append(temp[1])
        doc_id_list.append(temp[0])

word_freq = {}  # to remove rare words
for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_docs = []
for doc_content in doc_content_list:
    temp = clean_str(doc_content)
    words = temp.split()
    doc_words = []
    for word in words:
        # word not in stop_words and word_freq[word] >= 5
        if word not in stop_words and word_freq[word] >= 5:
            doc_words.append(word)

    doc_str = ' '.join(doc_words).strip()
    #if doc_str == '':
        #doc_str = temp
    clean_docs.append(doc_str)

final_clean_docs = [doc_id_list[i] + ' ' + clean_docs[i] for i in range(len(clean_docs))]
final_clean_corpus_str = '\n'.join(final_clean_docs)

with open('data/' + dataset + '.clean.txt', 'w') as f:
    f.write(final_clean_corpus_str)

min_len = 10000
aver_len = 0
max_len = 0

with open('data/' + dataset + '.clean.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        temp = line.split()
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)

aver_len = 1.0 * aver_len / len(lines)
print('Min_len : ' + str(min_len))
print('Max_len : ' + str(max_len))
print('Average_len : ' + str(aver_len))
