#!/usr/bin/env ipython

from tqdm import tqdm
import requests
import regex as re
import os
from pathlib import Path
os.chdir(os.path.dirname(__file__))
print(os.getcwd())

langs = ['en', 'fr'] #['de', 'en', 'es', 'fr', 'ja', 'zh']
dats = ['train', 'dev', 'test']

# ----------Amazon reviews data----------

dir = Path("Amazon reviews")
for dat in dats:
    for lang in langs:
        url = "https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/json/"+dat+"/dataset_"+lang+"_"+dat+".json"
        print(url)
        
        # Streaming, so we can iterate over the response.
        req = requests.get(url, stream=True)
        
        # Total size in bytes.
        total_size = int(req.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        
        # If file already exists
        if(os.path.isfile(dir + "/".join(re.split('/', url)[-2:]))): print("Already exists!!"); continue
        
        bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        #with open(os.path.join(re.split('/', url)[-2], re.split('/', url)[-1]), 'wb') as foo:
        with open(dir + "/".join(re.split('/', url)[-2:]), 'wb') as foo:
            for data in req.iter_content(block_size):
                bar.update(len(data))
                foo.write(data)
        bar.close()
        
        if total_size != 0 and bar.n != total_size: print("ERROR, something went wrong")
        
    print("done\n")


# ----------FB Research Muse pre-trained word embeddings---------

lang = ['en', 'fr'] #['en', 'fr', 'es', 'de']
urls = ['https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.'+lng+'.vec' for lng in lang]
print(urls)

dir = Path("bwe/vectors")
for url in urls:
    print(url)
    req = requests.get(url, stream=True)
    total_size = int(req.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    if(os.path.isfile(*re.split('/', url)[-1:])): print("Already exists!!"); continue
    
    bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(dir + re.split('/', url)[-1], 'wb') as foo:
        for data in req.iter_content(block_size):
            bar.update(len(data))
            foo.write(data)
    bar.close()
    
    if total_size != 0 and bar.n != total_size: print("ERROR, something went wrong")
    
print("done\n")