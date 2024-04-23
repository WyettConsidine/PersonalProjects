from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np 
import os
import re


def articleCorpusVectorizer(FolderDir, stemORlem = {'stemmer', 'lemmer', 'neither'}, maxFeatures = 30, max_df=1, min_df=1):
    filenameList = os.listdir(FolderDir)
    filenameList = [FolderDir+'/'+file for file in filenameList]
    #print(filenameList)
    if stemORlem == 'lemmer':
        LEMMER = WordNetLemmatizer()
        def lemFunc(str_input):
            words = re.sub(r'[^A-Za-z]',' ',str_input).lower().split()
            words= [LEMMER.lemmatize(word) for word in words]
            return words
        vectorizer = CountVectorizer(input = 'filename', 
                                     stop_words='english', 
                                     lowercase = True, 
                                     tokenizer = lemFunc,
                                     max_features = maxFeatures, 
                                     max_df=max_df,
                                     min_df=min_df )
    elif stemORlem == 'stemmer':
        STEMMER = PorterStemmer()
        def stemFunc(str_input):
            words = re.sub(r'[^A-Za-z]',' ',str_input).lower().split()
            words= [STEMMER.stem(word) for word in words]
            return words
        vectorizer = CountVectorizer(input = 'filename', 
                                     stop_words='english', 
                                     lowercase = True, 
                                     tokenizer = stemFunc,
                                     max_features = maxFeatures,
                                     max_df=max_df,
                                     min_df=min_df )
    else:
        vectorizer = CountVectorizer(input = 'filename', 
                                     stop_words='english', 
                                     lowercase = True, 
                                     max_features = maxFeatures, 
                                     max_df=max_df, 
                                     min_df=min_df )

    output = vectorizer.fit_transform(filenameList)
    vocab = vectorizer.get_feature_names_out()
    countVecDF = pd.DataFrame(output.toarray(), columns=vocab)
    for word in vocab:
        if any(char.isdigit() for char in word) | (len(word) <= 2):
            countVecDF= countVecDF.drop(columns= [word],)               
    
    return countVecDF, vectorizer


def contentVectorizer(filePath, maxFeatures = 30, inputOrigin = {'arXiv', 'newsAPI', 'scraped'}, stemORlem = {'stemmer', 'lemmer', 'neither'}, tfidf = False, sources = False, label = False, max_df=1, min_df=1):
    #read in and pre-process data
    if inputOrigin == 'arXiv':
        contentDF = pd.read_csv (filePath)
        contentR = contentDF['summary'].values[1:]
        if label:
            print(contentDF['Label'].values[1:])
            contentR = list(zip(contentDF['Label'].values[1:], contentR))
            contentR = [str(item[0])+','+str(item[1]) for item in contentR]
            #print(contentR)
    elif inputOrigin == 'newsAPI':
        with open(filePath) as file:
            contentR = [line.strip() for line in file]
    elif inputOrigin == 'scraped':
        with open(filePath) as file:
            contentR = [line.strip() for line in file]
    else:
        print("enter valid input origin")
        return None    
    
    if label:
        labels = [line.split(',', 1)[0] for line in contentR]
        #print(labels)
        content = [line.split(',', 1)[1] for line in contentR]
        #print(content)
    if sources:
        #print(content)
        labels = [line.split(',', 2)[0] for line in contentR]
        sources = [line.split(',', 2)[1] for line in contentR]
        print('here')
        content = [line.split(',', 2)[2] for line in contentR]
        #print(content)

    if stemORlem == 'lemmer':
        LEMMER = WordNetLemmatizer()
        def lemFunc(str_input):
            words = re.sub(r'[^A-Za-z]',' ',str_input).lower().split()
            words= [LEMMER.lemmatize(word) for word in words]
            return words
        vectorizer = CountVectorizer(input = 'content', 
                                     stop_words='english', 
                                     lowercase = True, 
                                     tokenizer = lemFunc, 
                                     max_features = maxFeatures, 
                                     max_df=max_df, 
                                     min_df=min_df )
    elif stemORlem == 'stemmer':
        STEMMER = PorterStemmer()
        def stemFunc(str_input):
            words = re.sub(r'[^A-Za-z]',' ',str_input).lower().split()
            words= [STEMMER.stem(word) for word in words]
            return words
        vectorizer = CountVectorizer(input = 'content', 
                                     stop_words='english', 
                                     lowercase = True, 
                                     tokenizer = stemFunc, 
                                     max_features = maxFeatures, 
                                     max_df=max_df, 
                                     min_df=min_df )
    elif tfidf == True:
        print('here in tfidf')
        LEMMER = WordNetLemmatizer()
        def lemFunc(str_input):
            words = re.sub(r'[^A-Za-z]',' ',str_input).lower().split()
            words= [LEMMER.lemmatize(word) for word in words]
            return words
        vectorizer = TfidfVectorizer(input = 'content', 
                                     stop_words='english', 
                                     lowercase = True, 
                                     tokenizer = lemFunc, 
                                     max_features = maxFeatures, 
                                     max_df=max_df, 
                                     min_df=min_df )
            
    else:
        vectorizer = CountVectorizer(input = 'content', 
                                     stop_words='english', 
                                     lowercase = True, 
                                     max_features = maxFeatures, 
                                     max_df=max_df, 
                                     min_df=min_df )

    output = vectorizer.fit_transform(content)
    vocab = vectorizer.get_feature_names_out()
    countVecDF = pd.DataFrame(output.toarray(), columns=vocab)
    print(len(countVecDF.columns))
    for word in vocab:
        if any(char.isdigit() for char in word)| (len(word) <= 3):
            countVecDF= countVecDF.drop(columns= [word],)
    if sources:
        countVecDF.insert(0, 'Source', sources, True)
    if label:
            countVecDF.insert(0, 'Label', labels, True) 
        

    
    print(len(countVecDF.columns))
    
    return countVecDF, vectorizer

def ArticleTfidfVectorizer(FolderDir, maxFeatures = 30):
    filenameList = os.listdir(FolderDir)
    filenameList = [FolderDir+'/'+file for file in filenameList]
    tfidfVect = TfidfVectorizer(input='filename', stop_words="english", max_features=maxFeatures)
    Vect = tfidfVect.fit_transform(filenameList)
    vocab = tfidfVect.get_feature_names_out()
    tfidfVecDF = pd.DataFrame(Vect.toarray(),columns=vocab)
    for word in vocab:
        if any(char.isdigit() for char in word) | (len(word) <= 2):
            countVecDF= countVecDF.drop(columns= [word],)               
    
    return tfidfVecDF, tfidfVect

def textProcessor(filepath,
                   textType={'content', 'corpus'},
                   contentOrigin = {'arXiv', 'newsAPI', 'scraped'}, 
                   stemORlem = {'stemmer', 'lemmer'}, 
                   labeled = False,
                   countORtfidf = {'count', 'tfidf'}, 
                   sources = False,
                   maxFeatures = 30, max_df=1, min_df=1):
    #Decision path to appropriate function
    if textType == 'content':
        if countORtfidf == 'tfidf':
            tfidfB = True
        else:
            tfidfB = False
        output, vectorizer = contentVectorizer(filepath,maxFeatures,contentOrigin,stemORlem,label=labeled,tfidf=tfidfB,
                                                sources=sources, max_df=max_df,min_df=min_df)
    elif textType == 'corpus':
        if countORtfidf == 'count':
            output, vectorizer = articleCorpusVectorizer(filepath, stemORlem, maxFeatures, max_df, min_df)
        elif countORtfidf == 'tfidf':
            output, vectorizer = ArticleTfidfVectorizer(filepath, maxFeatures)
        else:
            print('Select valid vectorization type. (count or tfidf)')
            output = None
            vectorizer = None
    else:
        print('Select valid text processing type.')
        output = None
        vectorizer = None
    return output, vectorizer


#This function takes in a file path containing raw data, and outputs a trnasactionalized form.
def makeTextTransactional(filePath, newFilePath):
    LEMMER = WordNetLemmatizer()     #Define a lemming tokenization function for Count Vect
    def lemFunc(str_input):
        words = re.sub(r'[^A-Za-z]',' ',str_input).lower().split()  #clean numerics, and make lowercase
        words= [LEMMER.lemmatize(word) for word in words]
        return words
    vectorizer = CountVectorizer(input = 'content',   #define count vectorizer, removing english stopwords,
                                    stop_words='english', 
                                    lowercase = True, 
                                    tokenizer = lemFunc, #use lemming
                                    max_features=100)  #Take top 100 most frequent words per document
    with open(filePath, 'r') as f:
        with open(newFilePath, 'w') as nf:
            for line in f:
                line = line.split(',',2)[2]   #remove labels and sources
                vectorizer.fit_transform([line]) 
                vocab = vectorizer.get_feature_names_out() #apply the count vectorizer, and take words
                words = list(vocab)
                words = [word for word in words if len(word) > 3] #remove words 3 char or less
                nf.write(','.join(words))
                nf.write('\n')  #write to new file
        nf.close()
    f.close()


def main():
    print('Text Processing Start')
    # outputArtCorpus, _ = textProcessor('./Assignment1/resourceFiles/corpus1(manual)', textType='corpus', countORtfidf='tfidf', stemORlem='lemmer')
    # print(outputArtCorpus)

    # outputarXiv, _ = textProcessor('resourceFiles\\corpus2(arXiv)\\arXivDataLabeled(query=Nuclear Energy).csv', textType='content', contentOrigin= 'arXiv', labeled = True, stemORlem='stemmer', maxFeatures=20, max_df=7, min_df=4)
    # print(outputarXiv)

    outputNewsAPI, _ = textProcessor('testDataLabeled.csv', textType='content', contentOrigin= 'newsAPI', labeled = True, stemORlem='lemmer', maxFeatures=30, max_df=100, min_df =10)
    print(outputNewsAPI.columns)
    
    #outputNewsAPI.to_csv('testDataLabeledDF')

    #outputScrape, _ = textProcessor('resourceFiles\\corpus4(bs4)\\webScrapedLabeled(query=Nuclear Energy)sustainability.csv', textType='content', contentOrigin= 'scraped', stemORlem='lemmer', labeled=True, maxFeatures=10, max_df = 15, min_df = 20)
    #print(outputScrape)


    # outputSt, dirVectorizer = articleCorpusVectorizer('./Assignment1/resourceFiles/corpus1(manual)', stemORlem='stemmer')
    # outputL, dirVectorizer = articleCorpusVectorizer('./Assignment1/resourceFiles/corpus1(manual)', stemORlem='lemmer')
    # print(outputSt.columns)
    # print(outputL.columns)

    
    # output2St, contVectorizer = contentVectorizer('./Assignment1/resourceFiles/corpus2(arXiv)/arXivData(query=nuclear energy)2024-01-30.csv', inputOrigin = 'arXiv', stemORlem='stemmer')
    # output2L, contVectorizer = contentVectorizer('./Assignment1/resourceFiles/corpus2(arXiv)/arXivData(query=nuclear energy)2024-01-30.csv', inputOrigin = 'arXiv', stemORlem='lemmer')
    # print(output2St.columns)
    # print(output2L.columns)

    # output3St, contVectorizer = contentVectorizer('./Assignment1/resourceFiles/corpus3(newsAPI)/newsapiData(query=nuclear energy)2024-01-30.csv', inputOrigin = 'newsAPI', stemORlem='stemmer')
    # output3L, contVectorizer = contentVectorizer('./Assignment1/resourceFiles/corpus3(newsAPI)/newsapiData(query=nuclear energy)2024-01-30.csv', inputOrigin = 'newsAPI', stemORlem='lemmer')
    # print(output3St.columns)
    # print(output3L.columns)
    #contentVectorizer('./Assignment1/resourceFiles/corpus3(newsAPI)/newsapiData(query=nuclear energy)2024-01-30.csv', inputOrigin = 'newsAPI', stemORlem='lemmer')


    # output4St, contVectorizer = contentVectorizer('./Assignment1/resourceFiles/corpus4(newsAPI)/newsapiData(query=nuclear energy)2024-01-30.csv', inputOrigin = 'newsAPI', stemORlem='stemmer')
    # output4L, contVectorizer = contentVectorizer('./Assignment1/resourceFiles/corpus4(newsAPI)/newsapiData(query=nuclear energy)2024-01-30.csv', inputOrigin = 'newsAPI', stemORlem='lemmer')
    # print(output4St.columns)
    # print(output4L.columns)

    # output = ArticleTfidfVectorizer('./Assignment1/resourceFiles/corpus1(manual)')
    # print(output)
    #makeTextTransactionalTest('testDataLabeled.csv','transactional.csv' )

    


if __name__ == "__main__":
    main()
    
    

    