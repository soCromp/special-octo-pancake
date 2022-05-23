# get keywords from wiki page associated with hypernym of each label
import pandas as pd
import spacy
nlp=spacy.load('en_core_web_md')
from string import punctuation
from re import sub
import nltk
# !pip3 install wikipedia-api
import wikipediaapi
import wikipedia


THRESHOLD=20

lemmatizer = nltk.WordNetLemmatizer()

def get_hotwords(text):
    text=sub("\[\d*\]", '', text)
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN'] # 1
    doc = nlp(text.lower()) # 2
    for token in doc:
        # 3
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        # 4
        if(token.pos_ in pos_tag):
            result.append(token.text)
                
    return result # 5


labels = [l.lower() for l in ['Landbird', 'Waterbird']]
wiki = wikipediaapi.Wikipedia('en')
out = []
lwords = set()

for l in labels:
    sub(' ', '_', l)
    page = wiki.page(l)
    if not page.exists(): 
        sugg = wikipedia.search(l, results=1)[0]
        page = wiki.page(sugg)
        print(l, 'has no wiki page! :( Using', sugg, 'instead')
    
    intro = page.summary
    w = [lemmatizer.lemmatize(k) for k in get_hotwords(page.summary)]
    lwords = lwords.union(set(w)).union( lemmatizer.lemmatize(l) )
    print(l, ':', w)
    out.append(','.join([l]+w)+'\n')


with open('labelkeys.csv', 'w') as f:
    f.write('label,keywords...\n')
    f.writelines(out)

#read in image caption keywords
path = 'words.csv'
cwords = [] #i-th element is set of words for i-th image
imgwords = [] #all words in array
with open(path, 'r') as f: 
    l=f.readline() #consume 1st line with headers
    for l in f:
        # print(l)
        l = l.split('.,')[1] # drop the caption itself
        l = l.strip().split(',') 
        l = [lemmatizer.lemmatize(w) for w in l]
        cwords.append(set(l))
        for w in l: imgwords.append(w)
print(cwords[:5])

ls = pd.Series(list(lwords))
cs = pd.Series(imgwords)
s = cs[~cs.isin(ls)]
s=s.value_counts() # get frequency of each word
print(s.shape)
print(s.to_string())

spur = s.index[s>THRESHOLD].to_list()
print(spur)
with open('spurious.txt', 'w+') as f:
    for w in spur:
        f.write(w+'\n')
