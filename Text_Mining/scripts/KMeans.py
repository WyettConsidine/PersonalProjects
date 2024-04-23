from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from textProcessing import textProcessor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score



def cleanEmptyLines(filePath):
    with open(filePath, 'r') as f:
        cleanedLines = []
        for line in f:
            if line.split(",", 1)[1] == '\n':
                continue
            else:
                cleanedLines.append(line)
        f.close()
    newFilePath = filePath.replace('.csv','')+' cleaned.csv'
    with open(newFilePath, 'w') as f:
        for line in cleanedLines:
            f.write(line)
        f.close()

#Read in the target datafile, and apply TFIDF, cleaning, and vocabularity reductions
def readInAndProcessText(filepath):
    print(f"Reading in {filepath}")
    # tfidfDF, _ = textProcessor(filepath,   #apply textProcessing function. See textProcessing.py script file. 
    #                                  textType='content', contentOrigin= 'scraped', stemORlem='neither', 
    #                                  countORtfidf='tfidf', sources = False, labeled=True, maxFeatures=15, max_df = .8, min_df = .3)

    tfidfDF, _ = textProcessor(filepath,   #apply textProcessing function. See textProcessing.py script file. 
                                    textType='content', contentOrigin= 'scraped', stemORlem='neither', 
                                    countORtfidf='tfidf', sources = False, labeled=True, maxFeatures=100, max_df = .9, min_df = .2)
    print(f"From File: {tfidfDF}")
    return tfidfDF

#seppartate the labels, sources, and numeric vectors from input tfidf dataframe
def stripLabelandSource(tfidfDF):
    labelsAndSources = []
    vectors = []
    vectorDim = len(tfidfDF.columns)-2
    for key, vals in tfidfDF.iterrows():   #itterate through the tfidf, appending source and lables to lists
        labelsAndSources.append((vals[0],vals[1]))
        vectors.append(vals[2:vectorDim])
    return labelsAndSources, vectors

#Take the numeric vectors, labels, and number of clusters to apply k-means
def applyKMeans(vectors, labelsASources, clusters):
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(vectors) #Apply K-means algorithm to inpu vectors
    clustLabels = kmeans.labels_   #Record cluster lables
    centers = kmeans.cluster_centers_ #record final cluster centroids

    lsClustered = list(zip(labelsASources,clustLabels)) #re-group labels, sources, and k-means partiton labels. 
    return lsClustered, centers

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


#Apply Silhouette Score to the Input TFIDF Dataframe
def silhouetteAnalysis(tfidfVects):
    Silhouette_Score =[]
    labelsASources, vectors = stripLabelandSource(tfidfVects) #Separate the Numeric vectors from the labels
    print(vectors)
    for i in range(2,7):
        lsClusters, centers = applyKMeans(vectors,labelsASources, i) #Apply K-means with [2-7] clusters
        labels = [x[1] for x in lsClusters]
        silScore = silhouette_score(vectors,labels)  #Calculate silhouette Score from Sklean.metrics
        Silhouette_Score.append(silScore)
    plt.plot([2,3,4,5,6], Silhouette_Score)
    plt.title("Silhouette Score Vs. Number of K-Means Clusters")    #Plot the Silhouette Scores from 2-7
    plt.xlabel("Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()

def readInHclusters(filePath):
    clusters = []
    with open(filePath, 'r') as f:
        for line in f:
            line.replace(' \n','')
            clusters.append([int(num)-1 for num in line.split(',')])
        f.close()
    return clusters

def compareKMeansHClust(hclust, kmeans):
    print(kmeans)
    print(hclust)
    print("here")
    simScores = []
    percSim1 = len(set(hclust[1]).intersection(set(kmeans[0])))/(len(set(hclust[1]).union(set(kmeans[0]))))
    print(percSim1)
    percSim2 = len(set(hclust[0]).intersection(set(kmeans[1])))/(len(set(hclust[0]).union(set(kmeans[1]))))
    print(percSim2)
    percSim3 = len(set(hclust[2]).intersection(set(kmeans[2])))/(len(set(hclust[2]).union(set(kmeans[2]))))
    print(percSim3)
    # percSim2 = len(set(hclust[i]) and set(kmeans[j])) / float(len(set(hclust[i]) or set(kmeans[j])))
    # percSim3 = len(set(hclust[i]) and set(kmeans[j])) / float(len(set(hclust[i]) or set(kmeans[j])))
    return simScores

def main():
    print("main")
    #cleanEmptyLines('resourceFiles\\corpus4(bs4)\\webScrapedLabeledSources(query=Nuclear Energy)risk(appendSet).csv')

    tfidfVects = readInAndProcessText('resourceFiles\\corpus4(bs4)\\webScrapedLabeledSources(query=Nuclear Energy)risk-LargeSet.csv')
    print(tfidfVects.columns)
    #tfidfVects.to_csv('resourceFiles\\tfidfData(100F).csv')
    # labelsASources, vectors = stripLabelandSource(tfidfVects)
    # lsClustered, centers = applyKMeans(vectors, labelsASources, 3)
    # LSCDF = LabeledToDF(lsClustered)
    # km = []
    # for df in list(LSCDF.groupby('Cluster Label')):
    #     km.append(list(df[1].index))
    # clusterlabels = []
    # for clustInds in km:
    #     labelRate = LSCDF.loc[clustInds]['Labels'].value_counts()
    #     print('Cluster Labels:')
    #     print(labelRate)
    #LSCDF.groupby()
    # clusterLabels = LSCDF['Cluster Label'].unique()
    # print(LSCDF.columns)
    # for cl in clusterLabels:
    #     print(f"Length of cluster {cl} = {len(LSCDF[LSCDF['Cluster Label'] == cl])}")
    # visClusters2D(centers, vectors, LSCDF)
    #silhouetteAnalysis(tfidfVects)


if __name__ == "__main__":
    main()
    