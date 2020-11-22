import os
import tempfile
import shutil
import urllib
import zipfile
import pandas as pd
import urllib.request
import json

import text_data

# Temporary folder for data we need during execution of this notebook (we'll clean up
# at the end, we promise)
temp_dir = os.path.join(tempfile.gettempdir(), 'mind')
os.makedirs(temp_dir, exist_ok=True)

# The dataset is split into training and validation set, each with a large and small version.
# The format of the four files are the same.
# For demonstration purpose, we will use small version validation set only.
base_url = 'https://mind201910small.blob.core.windows.net/release'
training_small_url = f'{base_url}/MINDsmall_train.zip'
validation_small_url = f'{base_url}/MINDsmall_dev.zip'
training_large_url = f'{base_url}/MINDlarge_train.zip'
validation_large_url = f'{base_url}/MINDlarge_dev.zip'

def download_url(url,
                 destination_filename=None,
                 progress_updater=None,
                 force_download=False,
                 verbose=True):
    """
    Download a URL to a temporary file
    """
    if not verbose:
        progress_updater = None
    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    if destination_filename is None:
        url_as_filename = url.replace('://', '_').replace('/', '_')
        destination_filename = \
            os.path.join(temp_dir,url_as_filename)
    if (not force_download) and (os.path.isfile(destination_filename)):
        if verbose:
            print('Bypassing download of already-downloaded file {}'.format(
                os.path.basename(url)))
        return destination_filename
    if verbose:
        print('Downloading file {} to {}'.format(os.path.basename(url),
                                                 destination_filename),
              end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)
    assert (os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    if verbose:
        print('...done, {} bytes.'.format(nBytes))
    return destination_filename

# For demonstration purpose, we will use small version validation set only.
# This file is about 30MB.
zip_path = download_url(training_small_url, verbose=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

os.listdir(temp_dir)

# The news.tsv file contains the detailed information of news articles involved in the behaviors.tsv file.
# It has 7 columns, which are divided by the tab symbol:
# - News ID
# - Category
# - Subcategory
# - Title
# - Abstract
# - URL
# - Title Entities (entities contained in the title of this news)
# - Abstract Entities (entities contained in the abstract of this news)
news_path = os.path.join(temp_dir, 'news.tsv')
df = pd.read_table(news_path,
              header=None,
              names=[
                  'id', 'category', 'subcategory', 'title', 'abstract', 'url',
                  'title_entities', 'abstract_entities'
              ])
os.makedirs('data', exist_ok=True)
news_json_path = os.path.join('data', 'news.json')
count = 0
with open(news_json_path, 'w') as outfile:
    for index, row in df.iterrows():
        if id is not None:
            print(count, row['title'])
            info = {
                'id': row['id'],
                'title': row['title'],
                'url': row['url'],
            }
            item = text_data.TextItem(info, row['title'])
            item.add_text(row['abstract'])
            if isinstance(row['title_entities'], str):
                entities = json.loads(row['title_entities'])
                for e in entities:
                    item.add_text(e['Label'])
            if isinstance(row['abstract_entities'], str):
                entities = json.loads(row['abstract_entities'])
                for e in entities:
                    item.add_text(e['Label'])
            for entry in item.get_entries():
                json.dump(entry, outfile)
                outfile.write('\n')
            count = count + 1

shutil.rmtree(temp_dir)

print('Data is ready at ', news_json_path)