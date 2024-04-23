This the repo for wyett considine's INFO 5871 homeworks. 

For Assignement 3: 
All Supervised models including NB, DT, and SVM are in the SupervisedMethods.py file located in ./scripts. 
Data used in this part of the assignement is pre-prcessed data called tfidfData.csv located in ./resourceFiles
Pictures/Trees/Diagrams used are in the figures directory.




For Assignment 2:
K-Means is in the K-means.py file.
Heirarchical Clustering is in the RAnalysis.rmd file
ARM is in the RAnalysis.rmd file
LDA is in the LDA.py file.
Data used is in the resroucesFile directory. 
Pictires of code, visualizations, and data in the figures directory. 



For Assignment 1:
Look in the Assignement1 folder.

There will be two directories, resourceFiles, and scripts. resrouceFiles contains all used data. It is organized into 4 corpuses, each with a description of how they were obtained. (bs4 is Beautiful Soup 4, a webscraping library.) The files that contain the word 'Labeled' have either a label at the end of the file name, such as 'safe', or 'risk'. If the 'Labeled' files do not have a label at the end, they are a combined file that contains both labels. The 4 labels used in the assigement are the words 'safe', 'risk', 'Sustainable', and 'Unsustainable'. The difference in data comes from changes to the Queries used to gather it. ie, data labeled with 'safe' contain the word 'safe' in the query, as discussed in class. 

The scripts directory has all runnable python code. This inscludes 2 api handling and cleaning scripts, a webscraping script, a text processing script, and an EDA script. The main function of each file has commented out lines that provided different funcitonality when un-commented. The text processing script is where the code is that does the following functions: Stemming, Lemmatization, CountVectorizer, TfidfVectorizor
