import xml.etree.ElementTree as ET
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import RegexpTokenizer




def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ' ' + ele
        # return string
    return str1

def preprocess(_w):
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(listToString(_w))
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    t = [lemmatizer.lemmatize(w) for w in _w if not w in stop_words]
    return t


def list_to_set(sentence):
    s = set()
    for w in sentence:
        s.add(w)
    return s

def wsd(term, sentence):
    context = list_to_set(sentence)
    bows = []
    bow = set()
    s = wn.synsets(term)
    max_sense = None
    for i in range(len(s)):
        bow = bow.union(preprocess(s[i].definition()))
        for e in s[i].examples():
            bow = bow.union(preprocess(e))
        bows.append((s[i],bow))
        bow = set()
    max_len = 0
    for i in range(len(bows)):
        overlap = bows[i][1] & context
        if len(overlap) > max_len:
            max_sense = bows[i][0]
            max_len = len(overlap)
    return max_sense

def semagram_extraction(sentence):
    semagram_tree = ET.parse('semagrams_300.xml')
    root = semagram_tree.getroot()
    semagram = set()
    synsets = set()
    for w in sentence:
        synset = wsd(w,sentence)
        if synset is None:
            continue
        else:
            synsets.add(synset)
    for s in synsets:
        for child in root:
            if child.attrib['synset'] == s.name():
                semagram.add(child)
    print(semagram)


i = 0
sentences = [s for s in brown.sents(categories='hobbies')]
for s in sentences:
    semagram_extraction(preprocess(s))