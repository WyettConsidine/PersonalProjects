import pandas as pd
import numpy as np
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.svm import SVC
import graphviz 

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split
# import tensorflow.keras
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import layers




import matplotlib.pyplot as plt


#Get TFIDF data from file
def grabData(filePath):
    tfidf = pd.read_csv(filePath, index_col = 0)
    #tfidf = tfidf.drop('Source', axis = 1)
    return tfidf

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
    #Split data into train/test
    TrainDF1, TestDF1 = train_test_split(dataDF, test_size=test_size)
    #Split labels from train/test sets
    noLabTest, LabTest = splitLabels(TestDF1)
    noLabTrain, LabTrain = splitLabels(TrainDF1)
    #return lists
    return noLabTrain, LabTrain, noLabTest, LabTest


############  Naive Bayes Section ###################
#Train a NB model with trianing data, and predict the labels. 
def trainNB(noLabTrain, trainLabs, noLabTest):
    #intstantiate NB
    MyModelNB = MultinomialNB()
    #fit data to NB
    NB = MyModelNB.fit(noLabTrain, trainLabs)
    #predict labels
    Preds = MyModelNB.predict(noLabTest)
    return NB, Preds

#######################################


###### Descision Tree Section ################
#train a DT model. Fit with training data, and return predicted labels
#max_depth 3 : 49.1
#no max depth: 46.1
#max depth = 5: 46.7
#
def trainDT(noLabTrain, trainLabs, noLabTest):
    #instatiate a DT object with input parameters
    DT=DecisionTreeClassifier(criterion='gini',
                            splitter='best',
                            max_depth=3, 
                            min_samples_split=4, 
                            min_samples_leaf=4, 
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

#Create a Visualizetion of the DT given the DT model
def dispTree(DT, tfidf, figName, saveImg = False):
    tree.plot_tree(DT)
    if saveImg:
        plt.savefig(figName)
    #Create a tree object with the DT model, input data, and parameters 
    dot_data = tree.export_graphviz(DT, out_file=None,
                    #The following creates TrainDF.columns for each
                    #which are the feature names.
                      feature_names=tfidf.columns,  
                      class_names=['safe','risk'],  
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
    clf = SVC(C=3, kernel="poly", degree = 3, verbose=False)
    #fit the model with the training data
    clf.fit(noLabTrain, trainLabs)
    #predict the test labels
    preds = clf.predict(noLabTest)
    return clf, preds

############################################

############ Nueral Networks: #################
def trainNN(noLabTrain, trainLabs, noLabTest, labTest):
    #Determine Common activiation function
    activationFunc = 'softplus'
    print(noLabTrain.shape[1])
    #Instantial the Keras NN
    My_NN_Model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, input_shape=(noLabTrain.shape[1],), activation=activationFunc),
    tf.keras.layers.Dense(4, activation=activationFunc), 
    tf.keras.layers.Dense(4, activation=activationFunc),  
    tf.keras.layers.Dense(4, activation=activationFunc),    
    tf.keras.layers.Dense(2, activation=activationFunc),  
    tf.keras.layers.Dense(1, activation='sigmoid') ## for 0 or 1
    ])
    My_NN_Model.summary()
    #Define the Loss function
    loss_function = keras.losses.BinaryCrossentropy(from_logits=False)
    My_NN_Model.compile(
                 loss=loss_function,
                 metrics=["accuracy"],
                 optimizer='adam'
                 )
    #Fit the data to the model, and define epochs
    Hist=My_NN_Model.fit(noLabTrain,trainLabs, epochs=300, validation_data=(noLabTest,labTest))
    #My_NN_Model.save("NuclearTextData_NN_Model.keras")
    return Hist, My_NN_Model


def testNN(My_NN_Model,noLabTest, labTest):
    #Use predict function to generate labels
    predictions=My_NN_Model.predict(noLabTest)
    print(predictions)
    print(type(predictions))
    #Encode predictions to num binary
    predictions[predictions >= .5] = 1
    predictions[predictions < .5] = 0
    print(predictions)
    labels = [0, 1]
    #Create a confusion matrix
    cm = confusion_matrix(y_pred=predictions, y_true=labTest, labels=labels)
    print(cm)
    #Use function to make a pretty confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=('Safe','Risk'))
    disp.plot()
    #adjust title
    plt.title(f'Confusion Matrix of Neural Network Model')
    #plot graphic
    plt.show()



#Get the averaged accuracy of input Sup Learning Model
#training func is the specified supervised learning model, ie trainNB()
#Normalize accuracy results over n tests. (default 20)
def getAvgAcc(tfidf, trainingFunc, tests = 20):
    sumAcc = 0
    #sum accuracy over n tests, each time getting a different train/test set
    for i in range(tests):
        noLabTrain, LabTrain, noLabTest, LabTest = splitData(tfidf)
        #train specified model
        _, preds = trainingFunc(noLabTrain, LabTrain, noLabTest)
        #add acc to running sumation
        sumAcc += accuracy_score(LabTest,preds)
    #normalize by num tests and return
    return sumAcc/tests


#Create a confusion matrix diagram. 
#input the data, the model name, and the model type
#If tests set to > 1, the model will average the confusion matrix entries
# over that number of tests. Not standard practice, but interesting to see.
#
def displayConfMat(tfidf, modelName, trainingFunction, tests = 1):
    accSum = 0
    for i in range(tests):
        #Split data into train/test labels/nonlabeled
        noLabTrain, LabTrain, noLabTest, LabTest = splitData(tfidf)
        #apply the data to the input training function.
        model, preds = trainingFunction(noLabTrain, LabTrain, noLabTest)
        #Using the true labels, and model predictions, make a confusion mat
        cm = confusion_matrix(LabTest, preds, labels= model.classes_)
        #If clause used in averaging across multiple tests
        if i == 0:
            cmSum = cm
        else:
            cmSum = np.add(cmSum,cm)
        accSum += accuracy_score(LabTest,preds)
    acc = accSum/tests
    cm = cmSum/tests
    #Use the ConfusionMatrixDisplay to load a visual
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=model.classes_)
    disp.plot()
    #adjust title
    plt.title(f'Confusion Matrix of {modelName} Model \n Accuracy = {np.round(acc,2)}')
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
    labelDict = {'safe':1,'risk':0}
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
    #sort the AUC results
    aucList.sort(key = lambda x: x[1],reverse = True)
    return aucList



#Code modified from SKLearn Documentaion Example code:
#https://scikit-learn.org/0.17/auto_examples/svm/plot_iris.html
def plotSVM(tfidf):
    X = tfidf[['nuclear','year']].values
    print(X)
    numDict = {'safe':1,'risk':0}
    y = [numDict[x] for x in tfidf['Label'].values]
    h = .02
    rbf_svc = SVC(kernel='rbf', degree=2, C=1).fit(X,y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = rbf_svc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired,edgecolors='black')
    plt.xlabel('Term "nuclear" Frequncy')
    plt.ylabel('Term "year" Frequncy')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('SVC with RBF kernel')

    plt.show()



if __name__ == '__main__' :
    tfidf = grabData('resourceFiles/tfidfData(100F).csv')
    
    #os.environ["PATH"] += os.pathsep + 'C:\\Users\\wyett\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python311\\site-packages\\graphviz'

    #print(tfidf.shape)
    #print(getAvgAcc(tfidf,trainDT,100))

    #NB model
    #displayConfMat(tfidf, 'Naive Bayes', trainNB)

    #DT Models:
    #displayConfMat(tfidf, 'Desicion Tree', trainDT)

    # ###Run only on WSL!!
    # noLabTrain, LabTrain, noLabTest, LabTest = splitData(tfidf)
    # DT, preds = trainDT(noLabTrain, LabTrain, noLabTest)
    # print(getAvgAcc(tfidf,trainDT,100))
    # dispTree(DT, noLabTrain, 'Decision Tree Model')

    ##SVM Models:
    #displayConfMat(tfidf, "Support Vector Machine", trainSVM)

    ### AUC:
    # modelNames = ['Naive Bayes', 'Desicion Tree', 'Support Vector Machine']
    # trainingFunctions = [trainNB,trainDT,trainSVM]
    # results= compareSupLearnMods(modelNames,trainingFunctions, tfidf)
    # print('AUC comparison across all supervised learning models:')
    # for result in results:
    #     print(f'Model: {result[0]}. AUC Score: {result[1]}')

    #plot SVM
    #plotSVM(tfidf)

    # #Run NN
    numDict = {'safe':1,'risk':0}
    tfidf['Label'] = [numDict[x] for x in tfidf['Label'].values]
    noLabTrain, LabTrain, noLabTest, LabTest = splitData(tfidf)
    hist, My_NN_Model = trainNN(noLabTrain,LabTrain,noLabTest,LabTest)
    print(hist)
    testNN(My_NN_Model,noLabTest,LabTest)
    dispNN(My_NN_Model)



