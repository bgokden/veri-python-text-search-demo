# veri python text search demo
Text Search Demo Using Veri And Universal Sentence Encoders

This repository intends to show how to prototype a semantic text search engine with Veri Feature Store.

It mainly uses:
* [Tensorflow Universal Sentence Encoders](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3)
* [Veri](https://github.com/bgokden/veri) as a Vector Search Engine
* [Microsoft News Recommendation Dataset](https://azure.microsoft.com/en-us/services/open-datasets/catalog/microsoft-news-dataset/)
* [spaCy](https://spacy.io/models/en#en_core_web_lg)
* [Pandas](https://pandas.pydata.org/)

## Requirments:
To use this example, you will need:
* Git
* Pip
* Python 3.7+
* An operating system that supports [Tensoflow](https://www.tensorflow.org/install). I tested everything on MacOS.

## Set up:
To start, git clone this repo and initialize environment for Unix:
```shell script
git clone git@github.com:bgokden/veri-python-text-search-demo.git
cd veri-python-text-search-demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
You can initialize your environment in a different way if you are using an IDE or Windows.
From now on, `python3` will be referred as `python`.


## Download and Prepare Dataset
`download.py` script will download the small training set of `Microsoft News Recommendation Dataset` 

and create `news.json` which includes Universal Sentence Vectors labeled with News id, title an URL.
```shell script
python download.py
```

This process can take a couple of hours depending on your computer.
Here we are using Microsoft News Recommendation Dataset partially.
News resource has id, url, title, abstract, title entities and abstract entities of news articles. Unfortunately, it doesn't include full articles due to copyright.

While creating local data:
An news article is split into sentences as:
* Title
* List of sentences in abstract
* list of title entity labels
* list of abstract entity labels

Similar to [Bag of Words Model](https://en.wikipedia.org/wiki/Bag-of-words_model), we define each article is a bag of sentences it includes.
A sentence is defined as one or more words. Thanks to Universal Sentence Encoders (USE), a word, ngrams and sentences can be mapped into same vector space.

Every sentence is encoded with USE and stored as group_label, label, feature.
Group label is metadata as json related to article, which is id, url, title.
Label is the text.
Feature is the 512 dimensional vector of the text encoded with USE.

## Upload dataset to Veri
When `download.py` is done you can start and upload the data into `veri` with `uploader.py`, it will automatically download and run a local veri instance. it will be downloaded to local `tmp` folder so you can delete this folder later.
```shell script
python uploader.py
```

This will take a couple of minutes. A pid file will be stored under `tmp` folder, you can kill the `veri` instance with this pid later.

Now you are ready to Search:
```python
import veriservice
from text_data import TextData

service = "localhost:5678"
client = veriservice.VeriClient(service, "news")

data = TextData(client)

res = data.search("Best movies")
print(res)
```
This example is also in `search_example.py`

#### Special Note:
__Please note that this is a single instance demo and this dataset is quite large so search can be slow, veri is designed to run in clusters which is not demonstrated here.__

This is an example search with default values:
```python
res = data.search("Best movies")
print(res)
res.head()
res[['title', 'url']].head()
```
Search result is a pandas dataframe so you can use dataframe tools to manipulate it.
Example result:
```python
      score                                              label  ...                                              title                                            url
0  1.712817                                      b'Film award'  ...  Roman Polanski Leads European Film Awards Nomi...  https://assets.msn.com/labs/mind/BBWvQFf.html
1  1.616612                                       b'Teen film'  ...  20 Teen Movies on Netflix Your Kids Will Love ...  https://assets.msn.com/labs/mind/BBUy2xG.html
2  1.567333                               b'Major film studio'  ...  Oscars: These Are the 42 Films Vying for Best ...  https://assets.msn.com/labs/mind/AAJDknJ.html
3  1.542766  b"All 112 of Netflix's notable original movies...  ...  All 112 of Netflix's notable original movies, ...  https://assets.msn.com/labs/mind/AAJIknr.html
4  1.358358                b"50 Best Movies You've Never Seen"  ...                   50 Best Movies You've Never Seen  https://assets.msn.com/labs/mind/AAHDxdZ.html
5  1.224054            b'The best football movies of all time'  ...               The best football movies of all time  https://assets.msn.com/labs/mind/AAI7lm0.html
6  1.145847                           b'Priceless (2016 film)'  ...           Talking husky says 'I love you' to owner  https://assets.msn.com/labs/mind/AAJNH1V.html
7  1.093771                           b'Hollywood Film Awards'  ...  From Al Pacino to Renee Zellweger, Fate Highli...  https://assets.msn.com/labs/mind/AAJO7s4.html
8  1.083639                  b'The most powerful LGBTQ movies'  ...                     The most powerful LGBTQ movies  https://assets.msn.com/labs/mind/AAIeIly.html
9  1.079842  b'7 great movies you can watch on Netflix this...  ...  7 great movies you can watch on Netflix this w...  https://assets.msn.com/labs/mind/AAHB9uG.html
```

Default search is using Cosine similarity as metric and using first 200 results to create groups using first 5 values to show first 10 results.
I will explain this values later.

If you use longer queries with multiple sentences they will searched separately and combined back again.
Veri has an internal cache for each query and it will be faster for the similar queries.

```python
res = data.search("Best movies", context=["I watch Netflix"])
print(res)
```