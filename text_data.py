import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_hub as hub
import tensorflow_text
import pandas as pd
import json
import spacy
from veriservice import VeriClient
from scipy.stats import entropy
import numpy as np

# python -m spacy download nl_core_news_lg
# python -m spacy download en_core_web_lg
nlp = spacy.load('en_core_web_lg')

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

entropy_threshold = 8.52

def escore_text(text):
    v = embed([text])[0]
    return entropy(np.abs(v), base=2)

def escore_feature(feature):
    return entropy(np.abs(feature), base=2)


def is_good_f(feature):
    return escore_feature(feature) >= entropy_threshold

def is_good(text):
    return escore_text(text) >= entropy_threshold


class TextItem:
    def __init__(self, info: str = None, text=None, split_threshold_min=20, split_threshold_max=100):
        self.info = info
        self.texts = []
        if text is not None:
            if isinstance(text, str):
                self.texts = [text]
            elif isinstance(text, list):
                self.texts = text
            else:
                self.texts = str(text)
        self.split_threshold_min = split_threshold_min
        self.split_threshold_max = split_threshold_max

    def add_text(self, text):
        if isinstance(text, str):
            self.texts.append(text.strip().rstrip("\n"))

    def calculate_texts(self):
        texts = []
        for text in self.texts:
            paragraphs = list(filter(lambda x: x != '', text.split('\n\n')))
            for paragraph in paragraphs:
                text = paragraph.replace("\n", " ").strip()
                if len(text) > self.split_threshold_max:
                    text_sentences = nlp(text)
                    sentences = []
                    prev = ""
                    for sentence in text_sentences.sents:
                        current = sentence.text
                        if len(prev) > 0:
                            current = prev + " " + current
                            prev = ""
                        if len(current) <= self.split_threshold_min or not is_good(current):
                            prev = current
                            continue
                        sentences.append(current.strip())
                    if len(prev) > 0:
                        sentences[-1] = sentences[-1] + " " + prev.strip()
                    texts.extend(sentences)
                else:
                    texts.append(text)
        self.texts = list(set(texts))

    def get_texts(self):
        return self.texts

    def get_info(self):
        return self.info

    def get_entries(self):
        self.calculate_texts()
        for text in self.texts:
            feature = embed([text])[0]
            yield {'label': text, 'group_label': self.info, 'feature': feature.numpy().tolist()}

    def get_features(self):
        for text in self.texts:
            yield embed([text])[0]


class TextData:
    def __init__(self,
                 client: VeriClient,
                 limit=200,
                 group_limit=5,
                 timeout=100000,
                 result_limit=10,
                 score_func_name="CosineSimilarity",
                 higher_is_better=True,
                 cache_duration=60,
                 prioritize_context=False):
        self.client = client
        self.limit = limit
        self.group_limit = group_limit
        self.timeout = timeout
        self.result_limit = result_limit
        self.score_func_name = score_func_name
        self.higher_is_better = higher_is_better
        self.prioritize_context = prioritize_context
        self.cache_duration = cache_duration

    def insert(self, item):
        for entry in item.get_entries():
            self.client.insert(entry['feature'], entry['label'].encode(), group_label=entry['group_label'].encode())

    def search(self, text, context=[], **kwargs):
        item_to_search = TextItem(text=text)
        item_to_search.calculate_texts()
        item_context = TextItem(text=context)
        item_context.calculate_texts()
        return self.item_search(item_to_search, item_context, **kwargs)

    def item_search(self, item, context, **kwargs):
        vectors = item.get_features()
        context_vectors = context.get_features()
        result = self.client.search(vectors,
                                    limit=kwargs.get('limit', self.limit),
                                    group_limit=kwargs.get('group_limit', self.group_limit),
                                    timeout=kwargs.get('timeout', self.timeout),
                                    score_func_name=kwargs.get('score_func_name', self.score_func_name),
                                    higher_is_better=kwargs.get('higher_is_better', self.higher_is_better),
                                    context_vectors=context_vectors,
                                    prioritize_context=kwargs.get('prioritize_context', self.prioritize_context),
                                    cache_duration=kwargs.get("cache_duration", self.cache_duration),
                                    result_limit=kwargs.get('result_limit', self.result_limit))
        results = []
        for r in result:
            group_label_data = json.loads(r.datum.key.groupLabel)
            results.append({
                'score': r.score,
                'label': str(r.datum.value.label),
                'group_label': group_label_data,
                'feature': r.datum.key.feature,
            })
        rs = pd.DataFrame(results)
        if 'group_label' in rs.columns:
            return pd.concat([rs.drop(['group_label'], axis=1), rs['group_label'].apply(pd.Series)], axis=1)
        return rs
