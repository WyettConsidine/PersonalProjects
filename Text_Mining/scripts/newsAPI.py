import requests  #to query the API 
import re  #regular expressions
import pandas as pd   # for dataframes
import random #For source selection
from datetime import date
from unidecode import unidecode
import csv
#Addapted from https://newsapi.org/docs/client-libraries/python

#List of sources and their polititcal lean:
#goal: approximate normal curve between sources of politically Left, left learning, center, right leaning, and right


# Init
api_key='efa4bfad3e8749d19ffd8298771cbfdb'
endpointArticles='https://newsapi.org/v1/articles'
endpointEverything='https://newsapi.org/v2/everything'
endpointSources = 'https://newsapi.org/v2/top-headlines/sources'

#random.seed(0)
# read in all relevant sources form the newapi 'sources' endpoint. 
# input endpoint (intended (endpointSources)), api key, and country. 
#output list of name-category pairs that are relevant to the discussion of nuclear energy.
#political lean of soruces determined with https://www.allsides.com/media-bias/ and https://mediabiasfactcheck.com/
def getSources(endpoint=endpointSources, key=api_key, country = 'us'):
    URLPost = {'apiKey': key,                #Define API key
                    #'category':'general',
                    'language': 'en',
                    'country':country,
                    }    
    respRaw = requests.get(endpoint,URLPost) #Make connection to API
    resp = respRaw.json()
    print(resp['status'])
    if resp['status'] != 'ok':   # test api connection status
        print("Non-Ok Status error from NewsApi. Status = " + resp['status'])
        return ['error']
    else:
        sources = []
        for src in resp['sources']:    #Itterate through sources
            if src['category'] in ('general', 'science', 'technology', 'health'): 
                #print(src['name'] + ", " + src['category'])
                sources.append((src['name'] , src['category'])) #Add name- category pairs if relevant
        return sources #return list of sources

# read in pre-recorded, relevant sources from last year from a csv file.
#the intended file was created with info from the 'getSources' funciton above.
#File in directory location ./Assignment1/resourceFiles/newsAPISourceLists.csv
#Input: nohting
#output: dicitonary mapping 5 possible political orientations to a list of news sources. 
def getSourceBiasLists():   
    with open('./Assignment1/resourceFiles/newsAPISourceLists.csv', newline='') as f: #read in sources
        reader = csv.reader(f)
        sourceFile = list(reader)
        f.close()
    sources = []
    lean = ['left', 'leftLean', 'center', 'rightLean', 'right']
    for srclist in sourceFile:   #create dictionary from the read in file.
        sources.append(srclist[1:]) #Foramt of the file = [polititcal lean, source1,source2,...]. Take all sources (ie, exclude first element.)
    newsSourceDict = dict(zip(lean, sources))  #Map polititcal leans as keys to the lists of sources as values. 
    #print(newsSourceDict)
    return newsSourceDict #return dictionary mapping poliitcal leans to lists of corresponding sources
        
#get a distribution of sources from the getSourcesBiasList dictionary.
#Purpose: Get a balanced set of sources to avoid bias in text information through random sampling.
#Get random int [1-3] to choose level of polititcal lean in sets of 2 sources. 
#To reduce political bias, we need to sample from oposite corresponsing levels of polititcal lean.
# 1 = center, center, 2 = lean left, lean right, 3 = left, right
#Then chose random sources form polititical lean category. 
def getDistributionOfSources(sourceBiasDict, numSources=10):
    pairs = int(numSources/2)
    remainder = numSources%2
    sources = []
    for i in range(0,pairs):
        setPair = random.randint(0,2)
        if setPair == 0:
            sources.append(sourceBiasDict['center'][random.randint(0, len(sourceBiasDict['center'])-1)])
            sources.append(sourceBiasDict['center'][random.randint(0, len(sourceBiasDict['center'])-1)])
        elif setPair == 1:
            sources.append(sourceBiasDict['leftLean'][random.randint(0, len(sourceBiasDict['leftLean'])-1)])
            sources.append(sourceBiasDict['rightLean'][random.randint(0, len(sourceBiasDict['rightLean'])-1)])
        else:
            sources.append(sourceBiasDict['left'][random.randint(0, len(sourceBiasDict['left'])-1)])
            sources.append(sourceBiasDict['right'][random.randint(0, len(sourceBiasDict['right'])-1)])
    if remainder == 1 :
        sources.append(sourceBiasDict['center'][random.randint(0, len(sourceBiasDict['left'])-1)])
    return sources


def basicNewsAPICall(endpoint, key, subject, numArticles=100):
    #define API call parameters
    URLPost = {'apiKey': key,
                    'language':'en',
                    'source': 'us', 
                    'pageSize': numArticles,
                    'sortBy' : 'relevancy',   #relevancy, popularity
                    'totalRequests': 20,
                    'q':subject}
    #Send API Call, and recienve the response
    print(URLPost)
    resp1 = requests.get(endpoint,URLPost)
    print(resp1.content)
    #return the response in JSON format
    return resp1.json()


def getArticleURLsFromSources(sourceList, key, endpoint, subject, numArticlesPerSource):
    articleURLS = []
    json = basicNewsAPICall(endpoint, key, subject, numArticlesPerSource)
    print(sourceList)
    for article in json['articles']:
        #print(article['source']['name'])
        if article['source']['name'] in sourceList:
            articleURLS.append(article['url']) 
    #print([art['url'] for art in json['articles']])
    #articleURLS.append([art['url'] for art in json['articles']])
    return articleURLS  

#Use function to bring parameters from this file to web scraping file
def getArticleURLsParams():
    return api_key, endpointEverything

def getArticleURLs(key, endpoint,subject, numArticles = 50, retSources = False):
    articleURLS = []
    #Use basic API call to get list of URLS
    json = basicNewsAPICall(endpoint, key, subject, numArticles)
    #Itterate through articles and collect URLS
    for article in json['articles']:
        if retSources:
            articleURLS.append((article['source']['name'], article['url']))
        else:
            articleURLS.append(article['url'])
    return articleURLS

def getArticleDesc(key, endpoint,subject, numArticles = 100):
    articleDesc = []
    #USE the basic API call
    json = basicNewsAPICall(endpoint, key, subject, numArticles)
    #Itterate through JSON, and collect article descriptions
    for article in json['articles']:
        articleDesc.append(article['description'])
    #Return list of descriptions
    return articleDesc

def strip_ascii(text):
    #Itterate through each character, keeping characters only in the ASCII range of acceptable text
    print(text)
    if text is not None:
        return "".join(
            char for char
            in text
            if 31 < ord(char) < 127
        )
    else:
        return ''

def articleDescToCSV(descList, subject):
    today = date.today()
    #Open a new CSV file, named with the query, and todays date
    with open(f'./Assignment1/resourceFiles/corpus3(newsAPI)/newsapiData(query={subject}){today}.csv', 'a') as f:
        for line in descList:
            print(strip_ascii(line))
            #Write the article Descriptions into the file, applying the cleaning function along the way
            f.write(f"{strip_ascii(line)}\n") 

def joinLabeledData(filePath1, l1, filePath2, l2, newFilePath):
    with (open(filePath1, 'r')) as f1:
        lines1 = [[l1, line.strip()] for line in f1]
        f1.close()
    with (open(filePath2, 'r')) as f2:
        lines2 = [[l2, line.strip()] for line in f2]
        f2.close()
    fullLines = lines1 + lines2
    with(open(newFilePath, 'w')) as newF:
        for line in fullLines:
            newF.write(line[0] + "," + line[1] + "\n")

def joinLabeledData4(filePath1, l1, filePath2, l2, filePath4, l4, filePath3, l3,newFilePath):
    with (open(filePath1, 'r')) as f1:
        lines1 = [[l1, line.strip()] for line in f1]
        f1.close()
    with (open(filePath2, 'r')) as f2:
        lines2 = [[l2, line.strip()] for line in f2]
        f2.close()
    with (open(filePath3, 'r')) as f3:
        lines3 = [[l3, line.strip()] for line in f3]
        f3.close()
    with (open(filePath4, 'r')) as f4:
        lines4 = [[l4, line.strip()] for line in f4]
        f4.close()
    fullLines = lines1 + lines2+lines3+lines4
    with(open(newFilePath, 'w')) as newF:
        for line in fullLines:
            newF.write(line[0] + "," + line[1] + "\n")


def get4topicsAndSave(topics, api_key, endpointEverything, newFilePath):
    flines = []
    for topic in topics:
        descList = getArticleDesc(api_key, endpointEverything, topic)
        labeled = [[topic, line.strip()] for line in descList]
        flines.append(labeled)
    fullLines = flines[0] + flines[1] + flines[2] + flines[3]
    with(open(newFilePath, 'w')) as newF:
        for line in fullLines:
            newF.write(line[0] + "," + line[1] + "\n")

#//////////////////////////////////// TEST ZONE
#MAKE THE BASIC CALL GRAB 100 FROM A TIMEFRAME, THEN WALK BACKWARDS UNTIL YOU HAVE N URL SOURCES

#json = basicNewsAPICall(endpointArticles, api_key,'Nuclear Energy', 'bbc-news')

#newsSourceDict = getSourceBiasLists()
#sources = getDistributionOfSources(newsSourceDict, numSources=20)
#res = getArticleURLsFromSources(sources, api_key, endpointEverything, 'Nuclear Energy', numArticlesPerSource=100)
#print(res)
        
#newsSourceDict = getSourceBiasLists()
#sources = getDistributionOfSources(newsSourceDict, numSources=6)
#print(sources)

#print(newsSourceDict['leftLean'])
#////////////////////////////////////

def main():
    #resp = basicNewsAPICall(endpointEverything,api_key,'technology')
    topics = ['technology','arts','media','science']
    newFilePath = 'testDataLabeled'


    # subject = topics[0]   
    # descList = getArticleDesc(api_key, endpointEverything, subject)
    # print(descList)
    # articleDescToCSV(descList, subject.replace('\"', ''))


    # subject = topics[1]       
    # descList = getArticleDesc(api_key, endpointEverything, subject)
    # print(descList)
    # articleDescToCSV(descList, subject.replace('\"', ''))


    # subject = topics[2]      
    # descList = getArticleDesc(api_key, endpointEverything, subject)
    # print(descList)
    # articleDescToCSV(descList, subject.replace('\"', ''))

    # subject = topics[3]       
    # descList = getArticleDesc(api_key, endpointEverything, subject)
    # print(descList)
    # articleDescToCSV(descList, subject.replace('\"', ''))
    
    # subject = '\"Nuclear Energy\" unsustainable'    
    # descList = getArticleDesc(api_key, endpointEverything, subject)
    # print(descList)
    # articleDescToCSV(descList, subject.replace('\"', ''))

    # joinLabeledData('resourceFiles\\corpus3(newsAPI)\\newsapiData(query=Nuclear Energy sustainable)2024-02-10.csv', 'sustainable',
    #                 'resourceFiles\\corpus3(newsAPI)\\newsapiData(query=Nuclear Energy unsustainable)2024-02-10.csv', 'unsustainable',
    #                 'resourceFiles\\corpus3(newsAPI)\\newsapiDataLabeled(query=Nuclear Energy)sustainability.csv')
    joinLabeledData4('resourceFiles\\corpus3(newsAPI)\\newsapiData(query=arts)2024-03-07.csv', 'arts',
                    'resourceFiles\\corpus3(newsAPI)\\newsapiData(query=media)2024-03-07.csv', 'media',
                    'resourceFiles\\corpus3(newsAPI)\\newsapiData(query=science)2024-03-07.csv', 'science',
                    'resourceFiles\\corpus3(newsAPI)\\newsapiData(query=technology)2024-03-07.csv', 'technology',
                    'testDataLabeled.csv')
       


    #print(getArticleURLs(api_key, endpointEverything, subject))


if __name__ == "__main__":
    main()