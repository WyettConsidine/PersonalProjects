
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from textProcessing import textProcessor
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import pyLDAvis
import pyLDAvis.lda_model


#######

#MyVectLDA_DH=CountVectorizer(input='filename')
##path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\SmallTextDocs"
#Vect_DH = MyVectLDA_DH.fit_transform(ListOfCompleteFiles)
#ColumnNamesLDA_DH=MyVectLDA_DH.get_feature_names()
#CorpusDF_DH=pd.DataFrame(Vect_DH.toarray(),columns=ColumnNamesLDA_DH)
#print(CorpusDF_DH)

######


def readInAndProcessText(filepath, raw=False):
    print(f"Reading in {filepath}")
    if raw == False:
        vectPD, vectorizer = textProcessor(filepath,
                                        textType='content', contentOrigin= 'scraped', stemORlem='lem', 
                                        countORtfidf='tfidf', sources = True, labeled=True, maxFeatures=100, max_df = 100, min_df = 10)
        print(f"From File: {vectPD}")
        return vectPD, vectorizer
    else:
        with open(filepath) as file:
            contentR = [line.strip() for line in file]
        labels = [line.split(',', 2)[0] for line in contentR]
        sources = [line.split(',', 2)[1] for line in contentR]
        print('here')
        content = [line.split(',', 2)[2] for line in contentR]
        LEMMER = WordNetLemmatizer()
        def lemFunc(str_input):
            words = re.sub(r'[^A-Za-z]',' ',str_input).lower().split()
            words= [LEMMER.lemmatize(word) for word in words]
            return words
        vectorizer = TfidfVectorizer(input = 'content', 
                                    stop_words='english', 
                                    lowercase = True, 
                                    tokenizer = lemFunc, 
                                    max_features = 100, 
                                    max_df=100, 
                                    min_df=10)
        MYdtm = vectorizer.fit_transform(content)
        return MYdtm,vectorizer

    


#Apply the LDA transformation.
def LDA(num_topics, countVectDF):
    lda_model_DH = LatentDirichletAllocation(n_components=num_topics, 
                                         max_iter=100, learning_method='online')  #construct the LDA obj

    LDA_DH_Model = lda_model_DH.fit_transform(countVectDF)  #apply the TFIDF
    return LDA_DH_Model, lda_model_DH  #return the results, and the LDA obj

## implement a print function 
## REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic:  ", idx)    
        print([(vectorizer.get_feature_names_out()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
                        ## gets top n elements in decreasing order



#Create a visual display of the topics, and their associated words. 
#Source: https://gatesboltonanalytics.com/
def plotLDA(lda_model_DH, vocab, num_topics):
    word_topic = np.array(lda_model_DH.components_)  #format input data
    word_topic = word_topic.transpose()
    num_top_words = 15
    vocab_array = np.asarray(vocab)
    fontsize_base = 20
    
    for t in range(num_topics):
        plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
        plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
        plt.xticks([])  # remove x-axis markings ('ticks')
        plt.yticks([]) # remove y-axis markings ('ticks')
        plt.title('Topic #{}'.format(t))
        top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
        top_words_idx = top_words_idx[:num_top_words]
        top_words = vocab_array[top_words_idx]
        top_words_shares = word_topic[top_words_idx, t]
        for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
            plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                    ##fontsize_base*share)
    plt.tight_layout()
    plt.show()

#Read in data, fit, plot, and visualize the topics using pyLDAvis.
def LDAVis(filepath, num_topics):
    mydtm, MyCountV = readInAndProcessText(filepath, raw=True) #Get data, and cv obj
    lda_mod = LatentDirichletAllocation(n_components=num_topics,max_iter=100, learning_method='online')#Get the lda obj
    lda_mod.fit_transform(mydtm)#fit to the LDA model
    print(type(lda_mod))
    panel = pyLDAvis.lda_model.prepare(lda_mod, mydtm, MyCountV) #create the LDA visualization dashboard
    pyLDAvis.save_html(panel, "LDAVis.html")


if __name__ == '__main__':
    num_topics = 3
    # countVectDF, MyCountV = readInAndProcessText('resourceFiles\\corpus4(bs4)\\weScrapedDataRFormat.csv')
    # lda_model_DH, lda_mod = LDA(num_topics,countVectDF.drop(['Label'], axis = 1).drop(['Source'], axis = 1))
    # print_topics(lda_mod, MyCountV, 15)
    # plotLDA(lda_mod, MyCountV.get_feature_names_out(), num_topics)

      #construct the LDA obj
    LDAVis('resourceFiles\\corpus4(bs4)\\weScrapedDataRFormat.csv', num_topics)
