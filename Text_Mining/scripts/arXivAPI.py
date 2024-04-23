import pandas as pd 
import requests
import pandas_read_xml as pdx
import xml.etree.ElementTree as ET
import re
from datetime import date

def basicArXivCall(subject, label, max_results=50):
    endpoint = "http://export.arxiv.org/api/query?"
    subject = subject.replace(" ", "+")
    print(subject)
    #Create Endpoint
    url = endpoint+"search_query=abs:"+subject+f'+AND+all:{label}'+f'&max_results={max_results}'
    print(url)
    try:                                                   #Attempt Connection
        requests.get(url)
    except requests.ConnectionError as e:
        print(e)
    else:
        print('API status: 200')
        with requests.get(url) as response:                   #upon successful connection  

            print(response.content)
            root = ET.fromstring(response.content)
            entries = []
            dictionary = {}
            for child in root.iter('*'):                     #Itterate through XML response
                tag = re.sub("\{.*?\}","",child.tag)                 #Basic regex cleaning
                print(tag)
                if tag == 'entry':         #If entering an 'entry' division/tag
                    entries.append(dictionary)#Add elements stored in 'dictionary'. 
                    print("dictionary " + str(dictionary)) #Dictionary populated in next if clause
                    dictionary = {}
                #If the tag/div is of the following, add the text field to the dictionary
                if tag in ['title','summary','primary_category']:
                    print(tag)
                    txt = child.text
                    if txt is not None:
                        txt = txt.replace(',','')
                    dictionary[tag] = txt

            print(entries)        
            df = pd.DataFrame(entries)
            if 'summary' in df.columns:                 #Clean 'summary' text if applicable
                print('df not empty')
                df['summary'] = df['summary'].apply(lambda x: str(x).replace('\n',' '))
                df['summary'] = df['summary'].apply(lambda x: str(x).replace('\t',' '))
                #df['label'] = label
            return df

            #df = pd.read_csv(path+'/arXiv_AI.txt',sep = '\t')        

def loadIntoFile(summaryDF, subject, label):
    today = date.today()
    summaryDF.to_csv(f'./Assignment1/resourceFiles/corpus2(arXiv)/arXivData(query={subject + " " +label}){today}.csv', index=False) 

def joinLabeledData(filePath1, l1, filePath2, l2, newFilePath):
    contentDF = pd.read_csv(filePath1)
    contentDF['Label'] = l1

    contentDF2 = pd.read_csv(filePath2)
    contentDF2['Label'] = l2

       
    frames = [contentDF, contentDF2]
    totalContent = pd.concat(frames)
    totalContent.to_csv(newFilePath, index=True)

def main():
     subject = '\"nuclear energy\"'
     datadf = basicArXivCall(subject, 'risk')
     #loadIntoFile(datadf, subject.replace('\"',''), 'risk')

    #  subject = '\"nuclear energy\"'
    #  datadf = basicArXivCall(subject, 'unsustainable')
    #  loadIntoFile(datadf, subject.replace('\"',''), 'unsustainable')

    # joinLabeledData('resourceFiles\\corpus2(arXiv)\\arXivData(query=nuclear energy risk)2024-02-08.csv', 'risk',
    #                 'resourceFiles\\corpus2(arXiv)\\arXivData(query=nuclear energy safe)2024-02-08.csv', 'safe',
    #                 'resourceFiles\\corpus3(newsAPI)\\arXivDataLabeled(query=Nuclear Energy).csv')

if __name__ == "__main__":
    main()