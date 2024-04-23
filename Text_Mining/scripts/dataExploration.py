import numpy as np
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

def readFile(filePath):
    with open(filePath, 'r') as f:
        text = f.read().replace('\n', '')
        f.close()
    return text


def basic_clean(string):
    string = unicodedata.normalize('NFKD', string)\
             .encode('ascii', 'ignore')\
             .decode('utf-8', 'ignore')
    string = re.sub(r'[^\w\s]', '', string).lower()
    return string

def tokenize(string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    string = tokenizer.tokenize(string, return_str = True)
    return string

def stem(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    string = ' '.join(stems)
    return string

def lemmatize(string):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(lemmas)    
    return string

def remove_stopwords(string, extra_words = [], exclude_words = []):
    stopword_list = stopwords.words('english')
    stopword_list = set(stopword_list) - set(exclude_words)
    stopword_list = stopword_list.union(set(extra_words))
    words = string.split()
    filtered_words = [word for word in words if (word not in stopword_list) & (len(word)>2)]
    string_without_stopwords = ' '.join(filtered_words)    
    return string_without_stopwords

def clean(text):
    return remove_stopwords(lemmatize(basic_clean(text)))

def wordCloud(text):
    print('making wordcloud')
    wc = WordCloud().generate(text)
    plt.imshow(wc)
    plt.show()

def createWordCloud(filepath):
    print('Making word cloud')
    text = readFile(filepath)
    text = clean(text)
    print(text)
    wordCloud(text)

if __name__ == "__main__":
    
    createWordCloud('resourceFiles\\corpus3(newsAPI)\\newsapiData(query=arts)2024-03-07.csv')
    
    createWordCloud('resourceFiles\\corpus3(newsAPI)\\newsapiData(query=media)2024-03-07.csv')

    createWordCloud('resourceFiles\\corpus3(newsAPI)\\newsapiData(query=science)2024-03-07.csv')

    createWordCloud('resourceFiles\\corpus3(newsAPI)\\newsapiData(query=technology)2024-03-07.csv')