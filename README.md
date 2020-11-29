# Veri Python Semantic Text Search Demo
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
As a side note, there is a data retention period which is 1 day by default. If you don't use a data for a day it will be deleted.


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
      score                                    label                                            feature      id                                              title                                            url
0  1.358358      b"50 Best Movies You've Never Seen"  [-0.010016842745244503, -0.05103179067373276, ...  N62924                   50 Best Movies You've Never Seen  https://assets.msn.com/labs/mind/AAHDxdZ.html
1  0.699587  b'The best football movies of all time'  [0.026803573593497276, -0.048604659736156464, ...  N23005               The best football movies of all time  https://assets.msn.com/labs/mind/AAI7lm0.html
2  0.685656        b'The 50 best films of the 2010s'  [-0.024614008143544197, 0.013537825085222721, ...   N6007                     The 50 best films of the 2010s  https://assets.msn.com/labs/mind/AAJAYsh.html
3  0.669703                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...  N48032  Movie review: Stars reunite for rom-com 'Todos...  https://assets.msn.com/labs/mind/AAGs9hb.html
4  0.669703                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...   N4855  This Boston Hotel Ranks As One of the Most Hau...  https://assets.msn.com/labs/mind/AAJhBGb.html
5  0.669703                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...  N22599              The 20 Most Haunted Hotels in America  https://assets.msn.com/labs/mind/AAI6Iey.html
6  0.669703                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...   N4912  New Movies and TV Shows You'll Be Able to Cozy...  https://assets.msn.com/labs/mind/AAJdRd0.html
7  0.668541     b'50 Best Movie Sequels of All Time'  [-0.03256732225418091, -0.03161001205444336, 0...  N26488                  50 Best Movie Sequels of All Time  https://assets.msn.com/labs/mind/BBWBrdA.html
8  0.659464                            b'Film award'  [-0.015458960086107254, 0.031246623024344444, ...  N20533  Roman Polanski Leads European Film Awards Nomi...  https://assets.msn.com/labs/mind/BBWvQFf.html
```

Default search is using Cosine similarity as metric and using first 200 results to create groups using first 5 values to show first 10 results.
I will explain this values later.

If you use longer queries with multiple sentences they will searched separately and combined back again.
Veri has an internal cache for each query and it will be faster for the similar queries.

```python
res = data.search("Best movies", context=["awards"])
print(res)
```
Search in the context of "awards" will prioritise best movies based on Film awards.
See the 9th result in the previous search is now the 2nd resullt.
```python
      score                                    label                                            feature      id                                              title                                            url
0  1.358358      b"50 Best Movies You've Never Seen"  [-0.010016842745244503, -0.05103179067373276, ...  N62924                   50 Best Movies You've Never Seen  https://assets.msn.com/labs/mind/AAHDxdZ.html
1  0.707995                            b'Film award'  [-0.015458960086107254, 0.031246623024344444, ...  N20533  Roman Polanski Leads European Film Awards Nomi...  https://assets.msn.com/labs/mind/BBWvQFf.html
2  0.699587  b'The best football movies of all time'  [0.026803573593497276, -0.048604659736156464, ...  N23005               The best football movies of all time  https://assets.msn.com/labs/mind/AAI7lm0.html
3  0.685656        b'The 50 best films of the 2010s'  [-0.024614008143544197, 0.013537825085222721, ...   N6007                     The 50 best films of the 2010s  https://assets.msn.com/labs/mind/AAJAYsh.html
4  0.669703                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...   N4855  This Boston Hotel Ranks As One of the Most Hau...  https://assets.msn.com/labs/mind/AAJhBGb.html
5  0.669703                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...  N22599              The 20 Most Haunted Hotels in America  https://assets.msn.com/labs/mind/AAI6Iey.html
6  0.669703                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...   N4912  New Movies and TV Shows You'll Be Able to Cozy...  https://assets.msn.com/labs/mind/AAJdRd0.html
7  0.669703                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...  N48032  Movie review: Stars reunite for rom-com 'Todos...  https://assets.msn.com/labs/mind/AAGs9hb.html
8  0.668541     b'50 Best Movie Sequels of All Time'  [-0.03256732225418091, -0.03161001205444336, 0...  N26488                  50 Best Movie Sequels of All Time  https://assets.msn.com/labs/mind/BBWBrdA.html
```
Context can be list of previous searchs, or list of article titles read by user.

If you have a system where context is more important than the actual search, there is prioritze_context parameter.

```python
res = data.search("Best movies", context=["awards"], prioritize_context=True)
print(res)
```
Same search but now the award is the 1st result.
```python
      score                                    label                                            feature      id                                              title                                            url
0  0.707995                            b'Film award'  [-0.015458960086107254, 0.031246623024344444, ...  N20533  Roman Polanski Leads European Film Awards Nomi...  https://assets.msn.com/labs/mind/BBWvQFf.html
1  0.398799                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...   N4912  New Movies and TV Shows You'll Be Able to Cozy...  https://assets.msn.com/labs/mind/AAJdRd0.html
2  0.398799                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...  N48032  Movie review: Stars reunite for rom-com 'Todos...  https://assets.msn.com/labs/mind/AAGs9hb.html
3  0.398799                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...  N62924                   50 Best Movies You've Never Seen  https://assets.msn.com/labs/mind/AAHDxdZ.html
4  0.398799                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...   N4855  This Boston Hotel Ranks As One of the Most Hau...  https://assets.msn.com/labs/mind/AAJhBGb.html
5  0.398799                                  b'Film'  [-0.021388016641139984, -0.016169555485248566,...  N22599              The 20 Most Haunted Hotels in America  https://assets.msn.com/labs/mind/AAI6Iey.html
6  0.203354        b'The 50 best films of the 2010s'  [-0.024614008143544197, 0.013537825085222721, ...   N6007                     The 50 best films of the 2010s  https://assets.msn.com/labs/mind/AAJAYsh.html
7  0.156346     b'50 Best Movie Sequels of All Time'  [-0.03256732225418091, -0.03161001205444336, 0...  N26488                  50 Best Movie Sequels of All Time  https://assets.msn.com/labs/mind/BBWBrdA.html
8  0.153158  b'The best football movies of all time'  [0.026803573593497276, -0.048604659736156464, ...  N23005               The best football movies of all time  https://assets.msn.com/labs/mind/AAI7lm0.html
```

You can also use filters to add some hard text matching:

```python
>>> data.search("Best movies", positive=["*Sequels*"])
      score                               label                                            feature      id                               title                                            url
0  0.668541   50 Best Movie Sequels of All Time  [-0.03256732225418091, -0.03161001205444336, 0...  N26488   50 Best Movie Sequels of All Time  https://assets.msn.com/labs/mind/BBWBrdA.html
1  0.550557  50 Worst Movie Sequels of All Time  [-0.023441022261977196, -0.005659815855324268,...  N27936  50 Worst Movie Sequels of All Time  https://assets.msn.com/labs/mind/BBWBrdB.html
>>> data.search("Best movies", positive=["*Sequels*"], negative=["dfds"])
      score                               label                                            feature      id                               title                                            url
0  0.668541   50 Best Movie Sequels of All Time  [-0.03256732225418091, -0.03161001205444336, 0...  N26488   50 Best Movie Sequels of All Time  https://assets.msn.com/labs/mind/BBWBrdA.html
1  0.550557  50 Worst Movie Sequels of All Time  [-0.023441022261977196, -0.005659815855324268,...  N27936  50 Worst Movie Sequels of All Time  https://assets.msn.com/labs/mind/BBWBrdB.html
>>> data.search("Best movies", positive=["*Sequels*"], negative=["*Worst*"])
      score                              label                                            feature      id                              title                                            url
0  0.668541  50 Best Movie Sequels of All Time  [-0.03256732225418091, -0.03161001205444336, 0...  N26488  50 Best Movie Sequels of All Time  https://assets.msn.com/labs/mind/BBWBrdA.html

```
Positive is using SQL like and negative is using SQL Not Like matching.


List of all parameters can be find in the `text_data.py`
Playing with different variables gives better results based on data type.

I will add more details in this demo and more explanation in architecture.

For questions please email me: berkgokden@gmail.com


