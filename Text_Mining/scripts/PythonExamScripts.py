from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import pyLDAvis.lda_model
import pyLDAvis
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import graphviz 
import unicodedata
import pandas as pd
import numpy as np 
import os
import re


########## Basic Data Preprocessing ####################
def compileFilePaths(fileDirPos,fileDirNeg):
    filenamesP = os.listdir(fileDirPos)
    filenamesN = os.listdir(fileDirNeg)
    filenamesList = [fileDirPos+'/'+file for file in filenamesP]
    filenamesList.extend([fileDirNeg+'/'+file for file in filenamesN])
    labels = []
    labels.extend(['Pos'] * len(filenamesP))
    labels.extend(['Neg'] * len(filenamesN))
    return filenamesList, labels

def countVectFunc(filenamesList, labels, max_features_cap = False, dontClean = False):
    LEMMER = WordNetLemmatizer()
    def lemFunc(str_input):
        words = re.sub(r'[^A-Za-z]',' ',str_input).lower().split()
        words= [LEMMER.lemmatize(word) for word in words]
        return words
    if max_features_cap:
        vectorizer = CountVectorizer(input = 'filename', 
                                stop_words='english', 
                                lowercase = True, 
                                tokenizer = lemFunc,
                                max_features = 100)
    else:
        vectorizer = CountVectorizer(input = 'filename', 
                                stop_words='english', 
                                tokenizer = lemFunc,
                                lowercase = True)
    output = vectorizer.fit_transform(filenamesList)
    if dontClean == False:
        vocab = vectorizer.get_feature_names_out()
        countVecDF = pd.DataFrame(output.toarray(), columns=vocab)
        for word in vocab:
            if any(char.isdigit() for char in word) | (len(word) <= 2):
                countVecDF= countVecDF.drop(columns= [word],)  
        countVecDF.insert(0, 'Label', labels, True)      
    else:
        countVecDF = output
    return countVecDF, vectorizer

def CleanDataFromRaw():
    fileNames, labels = compileFilePaths('resourceFiles/ExamCorpuses/Positive','resourceFiles/ExamCorpuses/Negitive')
    DF1, vect1 = countVectFunc(fileNames,labels)
    #DF1.to_csv('./resourceFiles/DFNoMaxFeat.csv')
    DFMF, vect2 = countVectFunc(fileNames,labels,True)
    #DFMF.to_csv('./resourceFiles/DFMaxFeat.csv')

##### Word cloud section ##########:

def readFile(filePath):
    with open(filePath, 'r',encoding="utf8") as f:
        text = f.read().replace('\n','')
        f.close()
    return text

def basic_clean(string):
    string = unicodedata.normalize('NFKD', string)\
             .encode('ascii', 'ignore')\
             .decode('utf-8', 'ignore')
    string = re.sub(r'[^\w\s]', '', string).lower()
    return string

def tokenize(string):
    tokenizer = ToktokTokenizer()
    string = tokenizer.tokenize(string, return_str = True)
    return string

def lemmatize(string):
    wnl = WordNetLemmatizer()
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

def wordCloud(text, title):
    print('making wordcloud')
    wc = WordCloud().generate(text)
    plt.imshow(wc)
    plt.title(title)
    plt.show()

def createWordCloud(filepath):
    print('Making word cloud')
    text = readFile(filepath)
    text = clean(text)
    wordCloud(text)

def compileTextToWCs():
    fileNames, labels = compileFilePaths('resourceFiles/ExamCorpuses/Positive','resourceFiles/ExamCorpuses/Negitive')
    posText = ''
    negText = ''
    pos = fileNames[:15]
    neg = fileNames[:30]
    for i in range(15):
        posText += str(readFile(pos[i]))
        negText += str(readFile(neg[i]))
    textP = clean(posText)
    textN = clean(negText)
    wordCloud(textP,'Postive Reviews')
    wordCloud(textN, 'Negitive Reviews')

###########################################################

####### Transactional Processing for ARM ###################
def readToSingleFile(posFileName,negFileName):
    fileNames, labels = compileFilePaths('resourceFiles/ExamCorpuses/Positive','resourceFiles/ExamCorpuses/Negitive')
    posText = ''
    negText = ''
    pos = fileNames[:15]
    neg = fileNames[15:30]
    for i in range(15):
        posText += str(readFile(pos[i])) + '\n'
        negText += str(readFile(neg[i])) + '\n'
    with open(posFileName, 'w') as f:
        for line in posText:
            f.write(line)
        f.close()

    with open(negFileName, 'w') as n:
        for line in negText:
            n.write(line)
        n.close()    

#This function takes in a file path containing raw data, and outputs a transactionalized form.
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
                #line = line.split(',',2)[2]   #remove labels and sources
                vectorizer.fit_transform([line]) 
                vocab = vectorizer.get_feature_names_out() #apply the count vectorizer, and take words
                words = list(vocab)
                words = [word for word in words if len(word) > 3] #remove words 3 char or less
                nf.write(','.join(words))
                nf.write('\n')  #write to new file
        nf.close()
    f.close()
########################################################



###### K-Means Section###########

#seppartate the labels, sources, and numeric vectors from input DF dataframe
def stripLabelandSource(DF):
    labels = []
    vectors = []
    vectorDim = len(DF.columns)-1
    for key, vals in DF.iterrows():   #itterate through the DF, appending source and labels to lists
        labels.append((vals[1]))
        vectors.append(vals[2:vectorDim])
    return labels, vectors

#Take the numeric vectors, labels, and number of clusters to apply k-means
def applyKMeans(vectors, labels, NumClusters):
    kmeans = KMeans(n_clusters=NumClusters, random_state=0, n_init="auto").fit(vectors) #Apply K-means algorithm to inpu vectors
    clustLabels = kmeans.labels_   #Record cluster lables
    centers = kmeans.cluster_centers_ #record final cluster centroids

    lsClustered = list(zip(labels,clustLabels)) #re-group labels, sources, and k-means partiton labels. 
    return lsClustered, centers

def testClusters(lsClustered):
    pos1 = 0
    neg1 = 0
    for pair in lsClustered:
        if (pair[0] == 'Pos') & (pair[1] == 1):
            pos1 += 1
        if (pair[0] == 'Neg') & (pair[1] == 1):
            neg1 += 1
    if pos1 > neg1:
        accuracy = (pos1 + (15-neg1))/30
    else:
        accuracy = (neg1 + (15-pos1))/30 
    return accuracy

#convert labels, sources, and cluster labels into a dataframe
def LabeledToDF(lsClustered):
    labels = []
    sources = []
    cluster = []
    for point in lsClustered:
        labels.append(point[0][0])
        sources.append(point[0][1])
        cluster.append(point[1])
    df = pd.DataFrame({'Labels':labels, 'Sources':sources, 'Cluster Label': cluster})

    return df

#Take the input dataframe with labels, clusters and sources.
#Apply pca to the numeric vectors to define top 2 variance capturing dimensions.
#Fit the vectors to the newly oriented axis, reducing to 2 dimensions.
#Plot the clustered based on the k-means label, as well as the centroids. 
def visClusters2D(centers, vectors, LSCDF):
    print("visualizations")
    pca = PCA(n_components=2)  #Apply pca
    points = pca.fit_transform(vectors)
    pcaCenters = pca.transform(centers) #Reduce the vectors to 2 dim
    print(points)
    LSCDF['x'] = [x[0] for x in points]
    LSCDF['y'] = [x[1] for x in points]
    print(LSCDF)
    unique_labels = LSCDF['Cluster Label'].unique() 
    colors = ['y', 'c', 'm', 'r', 'g', 'b']
    for i, label in enumerate(unique_labels):   #plot the clusters by color, based on k-means cluster
        cluster_data = LSCDF[LSCDF['Cluster Label'] == label] 
        plt.scatter(cluster_data['x'], cluster_data['y'], c=colors[i], label=f"Cluster {label}")
    pcaCentersX = [x[0] for x in pcaCenters]
    pcaCentersY = [x[1] for x in pcaCenters] 
    plt.scatter(x=pcaCentersX, y=pcaCentersY, color = 'black', label='K-Means Centroids') #plot centroids
    plt.legend(loc="upper left")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title(f"2D Cluster Projections to Analyze K-Means Partitions \n Number of Clusters: {len(unique_labels)}")
    plt.show()
##################################################

######### LDA Section ##########################
#Apply the LDA transformation.
def LDA(num_topics, countVectDF):
    lda_model_DH = LatentDirichletAllocation(n_components=num_topics, 
                                         max_iter=100, learning_method='online')  #construct the LDA obj

    LDA_DH_Model = lda_model_DH.fit_transform(countVectDF)  #apply the TFIDF
    return LDA_DH_Model, lda_model_DH  #return the results, and the LDA obj

#Use the LDA model to plot the top terms for each generated topic
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
        top_words = [word for word in top_words if len(word) > 2]
        top_words_shares = word_topic[top_words_idx, t]
        for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
            plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                        ##fontsize_base*share)
    plt.tight_layout()
    plt.show()

#####################################

######## Supervised Learning Data Preparation Section ####################

#Remove, but keep lables. Separate into 2 lists
def splitLabels(trainDF1):
    trainLabs = trainDF1['Label']
    trainDF1 = trainDF1.drop('Label', axis = 1)
    return trainDF1, trainLabs

#Separate training and test data based on test_size.
#(test_size = .2 would separate 80% into the training set, and 20% into test set)
#Split the data into 4 parts:
#    Training data without labels  (noLabTrain)
#    Labels of training set  (LabTrain)
#    Test data without labels (noLabTest)
#    Labels of test set  (LabTest)
# Return 4 parts
def splitData(dataDF, test_size = .2):
    # print("Original dimensions of input dataframe: ", dataDF.shape)
    # print("-------------------------------")
    #Split data into train/test
    TrainDF1, TestDF1 = train_test_split(dataDF, test_size=test_size)
    # print("Dimensions of training dataframe: ", TrainDF1.shape)
    # print("Dimensions of testing dataframe: ", TestDF1.shape)
    # print("-------------------------------")
    #Split labels from train/test sets
    TrainData, TrainLabels = splitLabels(TrainDF1)
    TrainData.drop('Unnamed: 0', axis = 1, inplace=True)
    # print("First Entries in Training Data: \n", TrainData.head(3))
    # print("First Entries in Training Labels: \n", TrainLabels.head(3))
    # print("-------------------------------")
    TestData, TestLabels = splitLabels(TestDF1)
    TestData.drop('Unnamed: 0', axis = 1, inplace=True)
    # print("First Entries in Testing Data: \n", TestData.head(3))
    # print("First Entries in Testing Labels: \n", TestLabels.head(3))
    # print("-------------------------------")
    #return lists
    return TrainData, TrainLabels, TestData, TestLabels
########################################################

############  Naive Bayes Section ###################
#Train a NB model with trianing data, and predict the labels. 
def trainNB(noLabTrain, trainLabs, noLabTest):
    #intstantiate NB
    MyModelNB = MultinomialNB()
    #fit data to NB
    NB = MyModelNB.fit(noLabTrain, trainLabs)
    #predict labels
    Preds = MyModelNB.predict(noLabTest)
    return MyModelNB, Preds
#######################################


########### Decision Tree Models ########################
def trainDT(noLabTrain, trainLabs, noLabTest):
    #instatiate a DT object with input parameters
    DT=DecisionTreeClassifier(criterion='gini',
                            splitter='best',
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            class_weight=None)
    #fit DT obj to training data
    DT.fit(noLabTrain, trainLabs)
    #predict test labels
    Preds = DT.predict(noLabTest)
    return DT, Preds

#Create a Visualization of the DT given the DT model
def dispTree(DT, tfidf, figName, saveImg = False):
    tree.plot_tree(DT)
    if saveImg:
        plt.savefig(figName)
    #Create a tree object with the DT model, input data, and parameters 
    dot_data = tree.export_graphviz(DT, out_file=None,
                    #The following creates TrainDF.columns for each
                    #which are the feature names.
                      feature_names=tfidf.columns,  
                      class_names=['Negative','Positive'],  
                      filled=True, rounded=True,  
                      special_characters=True)
    #Use graphviz to plot the tree object                             
    graph = graphviz.Source(dot_data) 
    ## Create dynamic graph name
    tempname=str(figName)
    #render the tree
    graph.render(tempname) 
##############################################


#### Support Vector Machines ################
#create and train SVM. Use to predict test labels
def trainSVM(noLabTrain, trainLabs, noLabTest):
    #create SVM model with soft margins, cost = C
    clf = SVC(C=1, kernel="rbf", degree = 3, verbose=False)
    #fit the model with the training data
    clf.fit(noLabTrain, trainLabs)
    #predict the test labels
    preds = clf.predict(noLabTest)
    return clf, preds

############################################



################# Supervised Learning Evaluation Code ###############

#Get the averaged accuracy of input Sup Learning Model
#training func is the specified supervised learning model, ie trainNB()
#Normalize accuracy results over n tests. (default 20)
def getAvgAcc(tfidf, trainingFunc, tests = 30):
    sumAcc = 0
    #sum accuracy over n tests, each time getting a different train/test set
    for i in range(tests):
        TrainData, TrainLabels, TestData, TestLabels = splitData(tfidf)
        #train specified model
        _, preds = trainingFunc(TrainData, TrainLabels, TestData)
        #add acc to running sumation
        sumAcc += accuracy_score(TestLabels,preds)
    #normalize by num tests and return
    return sumAcc/tests


#Create a confusion matrix diagram. 
#input the data, the model name, and the model type

def displayConfMat(tfidf, modelName, trainingFunction):
    #Split data into train/test labels/nonlabeled
    TrainData, TrainLabels, TestData, TestLabels = splitData(tfidf)
    #apply the data to the input training function.
    model, preds = trainingFunction(TrainData, TrainLabels, TestData)
    #Using the true labels, and model predictions, make a confusion mat
    cm = confusion_matrix(TestLabels, preds, labels= model.classes_)
    #Use the ConfusionMatrixDisplay to load a visual
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=model.classes_)
    disp.plot()
    #adjust title
    plt.title(f'Confusion Matrix of {modelName} Model \n No Max Features')
    #plot graphic
    plt.show()

#compare each supervised learning model averaged over a number of tests using AUC
#input the model names, the corresponding training functions, the raw data,
# and the number of tests to average over. 
# Over each test, split the data, and train each model on it. 
#For each model, predict the labels, calculate ROC and AUC,
# then add to the running totals. 
#At the end, average across the number of tests.
def compareSupLearnMods(modNames, trainingFunctions, tfidf, numTests = 20):
    #instantiate the running sum list
    aucList = []
    #dictionary for propper formatting
    labelDict = {'Pos':1,'Neg':0}
    #for each test
    for t in range(numTests):
        predTrueLabLists = []
        #Train the models on a test/training split.
        for model in trainingFunctions:
            #split data into test/train label/non-labled
            noLabTrain, LabTrain, noLabTest, LabTest = splitData(tfidf)
            #predict and record test labels.
            _, preds = model(noLabTrain, LabTrain, noLabTest)
            predTrueLabLists.append([LabTest.values, preds])
        #For each model,
        for i in range(len(modNames)):
            #retrieve test labels, and use dictionary for propper formatting
            y = [labelDict[l] for l in predTrueLabLists[i][0]]
            pred = [labelDict[l] for l in predTrueLabLists[i][1]]
            #Calculate ROC
            fpr, tpr, thresholds = roc_curve(y,pred)
            if t == 0:  #If it is the first test, just populate the running sum list.
                aucList.append((modNames[i], auc(fpr, tpr)))
            else:
                #Add the AUC to the running total for that model across the tests
                aucList[i] = (aucList[i][0], aucList[i][1] + auc(fpr, tpr))
    #normalize by number of tests
    aucList = [(aucVal[0],aucVal[1]/t) for aucVal in aucList]
    print(aucList)
    #sort the AUC results
    aucList.sort(key = lambda x: x[1],reverse = True)
    return aucList





if __name__ == '__main__':
    print("Beginning Analysis ... ")
    DFNoMaxF = pd.read_csv('resourceFiles/DFNoMaxFeat.csv')
    DFMaxF = pd.read_csv('resourceFiles/DFMaxFeat.csv')

    # # Process the data into transactional format for ARM
    # readToSingleFile('resourceFiles\\ExamCorpuses\\positiveReviews.txt','resourceFiles\\ExamCorpuses\\negitiveReviews.txt')
    # makeTextTransactional('resourceFiles\\ExamCorpuses\\positiveReviews.txt','resourceFiles\\ExamCorpuses\\posRevsBasket.csv' )
    # makeTextTransactional('resourceFiles\\ExamCorpuses\\negitiveReviews.txt','resourceFiles\\ExamCorpuses\\negRevsBasket.csv' )

    # # Cluster Visualization Section
    # labels, vectors = stripLabelandSource(DFMaxF)
    # clusters, centers = applyKMeans(vectors, labels,2)
    # TempDF = LabeledToDF(clusters)
    # visClusters2D(centers, vectors, TempDF)
    # print(testClusters(clusters))

    # #LDA:
    # fileNames, labels = compileFilePaths('resourceFiles/ExamCorpuses/Positive','resourceFiles/ExamCorpuses/Negitive')
    # countVectDF, vect1 = countVectFunc(fileNames,labels, dontClean=True,max_features_cap=True)
    # num_topics = 2
    # lda_model_DH, lda_mod = LDA(num_topics,countVectDF)
    # plotLDA(lda_mod, vect1.get_feature_names_out(), num_topics)

    # # supervised learning data split
    # splitData(DFMaxF, test_size = .2)

    # # Supervised Learning Models:
    #displayConfMat(DFNoMaxF, 'Naive Bayes', trainNB)
    #displayConfMat(DFNoMaxF, 'Decision Tree', trainDT)
    #displayConfMat(DFNoMaxF, 'Support Vector Machine', trainSVM)

    # ###Render and Display Graph
    # noLabTrain, LabTrain, noLabTest, LabTest = splitData(DFMaxF)
    # DT, preds = trainDT(noLabTrain, LabTrain, noLabTest)
    # dispTree(DT, noLabTrain, 'Decision Tree Model')

    ## # AUC Comparisons:
    modelNames = ['Naive Bayes', 'Desicion Tree', 'Support Vector Machine']
    trainingFunctions = [trainNB,trainDT,trainSVM]
    results= compareSupLearnMods(modelNames,trainingFunctions, DFNoMaxF, numTests=20)
    print('AUC comparison across all supervised learning models:')
    for result in results:
        print(f'Model: {result[0]}. AUC Score: {result[1]}')