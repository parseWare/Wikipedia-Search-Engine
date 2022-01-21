import re
import os
import sys
import time
import math
import pickle
from Stemmer import Stemmer
from nltk.corpus import stopwords
from collections import defaultdict,OrderedDict

titles_dict = defaultdict(dict)
index_dict = defaultdict(dict)
check_dict = {'1':("a","b","c"),
              '2':("d","e","f"),
              '3':("g","h","i"),
              '4':("j","k","l"),
              '5':("m","n","o"),
              '6':("p","q","r"),
              '7':("s","t","u"),
              '8':("v","w","x"),
              '9':("y","z"),
              '10':("0","1","2","3","4","5","6","7","8","9")}
fields= ['t:', 'i:', 'c:', 'b:', 'r:', 'l:']
documentLimit = 50000
stemmer = Stemmer("english")
stop_words = set(stopwords.words('english'))

titles_path = "./titles/title_{}"
inverted_index_path = "./indexes/final_{}"
output_file_path = "./queries_op.txt"
queries_path = "./queries.txt"

def tokenization(content):
    tokens = re.findall("\d+|[\w]+",content)
    tokens = [t for t in tokens]
    return tokens

def preprocess(data):
    return [stemmer.stemWord(token.casefold()) for token in tokenization(data) if token and token.casefold() not in stop_words]

def getIndexFile(word):
    if word[0] in check_dict['1']:
        return 1
    elif word[0] in check_dict['2']:
        return 2
    elif word[0] in check_dict['3']:
        return 3
    elif word[0] in check_dict['4']:
        return 4
    elif word[0] in check_dict['5']:
        return 5
    elif word[0] in check_dict['6']:
        return 6
    elif word[0] in check_dict['7']:
        return 7
    elif word[0] in check_dict['8']:
        return 8
    elif word[0] in check_dict['9']:
        return 9
    elif word[0] in check_dict['10']:
        return 10
    else:
        return -1

def readTitles(num):
    file = open(titles_path.format(str(num)),"rb")
    titles = pickle.load(file)
    return titles

def loadTitles():
    global titles_dict
    path, dirs, files = next(os.walk("./titles"))
    file_count = len(files)
#     print(file_count)
    count =0
    for i in range(1,file_count+1):
        titles_dict[i] = readTitles(i)
        count += len(titles_dict[i])
    return count
# print(TOTALDOCS)

def readIndex(num):
    file = open(inverted_index_path.format(str(num)),"rb")
    index = pickle.load(file)
    return index

def calPlainFrequency(posting_list):
    Fields= ['t', 'b', 'i', 'c', 'r', 'l']
    
    fieldWeights= [0.75, 0.25, 0.15, 0.12, 0.12, 0.12]
    
    numOfOccurences= 0
    total_count=0
    for f in range (0, len(Fields)):
        index= posting_list.find(Fields[f])
        if index !=-1:
            index +=1
            count =''
            while index< len(posting_list) and posting_list[index].isdigit():
                count += posting_list[index]
                index +=1
            numOfOccurences += int(count) * fieldWeights[f]
        
    return numOfOccurences

def tfidfPlain(wordPostings):
    tfidfScores ={}
    for word in wordPostings.keys():
        posting_lists= wordPostings[word].split('|')
        numOfDocs= len(posting_lists)
        totalDocs= TOTALDOCS
        for pl in range(0, numOfDocs):
            if posting_lists[pl] != '':
                pageID= posting_lists[pl].split('#')[0][1:]
                numOfOccurences = calPlainFrequency(posting_lists[pl].split('#')[1])
                tf = 1 + math.log(numOfOccurences)
                idf= totalDocs/(numOfDocs+1)
                tfidf= tf * math.log(idf)
                
                if pageID not in tfidfScores:
                    tfidfScores[pageID]= tfidf
                else:
                    tfidfScores[pageID] +=tfidf
      
    return tfidfScores

def calFieldFrequency(posting_list, current_field):
    Fields= ['t', 'b', 'i', 'c', 'r', 'l']
    fieldWeights= [0.90, 0.25, 0.15, 0.12, 0.12, 0.12]
    
    numOfOccurences= 0
    flag= False
    index= posting_list.find(current_field)
    if index !=-1:
        flag =True
        index +=1
        count =''
        while index< len(posting_list) and posting_list[index].isdigit():
            count += posting_list[index]
            index +=1
        numOfOccurences += int(count)* fieldWeights[Fields.index(current_field)]
        
        
    if flag ==False:   
        for f in range (0, len(Fields)):
            index= posting_list.find(Fields[f])
            if index !=-1:
                index +=1
                count =''
                while index< len(posting_list) and posting_list[index].isdigit():
                    count += posting_list[index]
                    index +=1
                numOfOccurences += int(count) *  fieldWeights[f]
    return numOfOccurences

def tfidfField(wordPostings):
    tfidfScores ={}
    for word in wordPostings.keys():
        current_field= wordPostings[word][1]
        posting_lists= wordPostings[word][0].split('|')
        numOfDocs= len(posting_lists)
        totalDocs= TOTALDOCS

        for pl in range(0, numOfDocs):
            if posting_lists[pl] != '':
                pageID= posting_lists[pl].split('#')[0][1:]
                numOfOccurences= calFieldFrequency(posting_lists[pl].split('#')[1], current_field)
                tf = 1 + math.log(numOfOccurences)
                idf= totalDocs/(numOfDocs+1)
                tfidf= tf * math.log(idf)

                if pageID not in tfidfScores:
                    tfidfScores[pageID]= tfidf
                else:
                    tfidfScores[pageID] +=tfidf
        
    return tfidfScores

def getPostingList(fileWordMap):
    global index_dict
    wordPostings= {}
    for file_num, words in fileWordMap.items():
        for w in words:
            if w in index_dict[file_num]:
                wordPostings[w]= index_dict[file_num][w]
            else:
                wordPostings[w]= ''
    return wordPostings

def getPostingListField(fileWordMap):
    wordPostings= {}
    for file_num, entry in fileWordMap.items():
        for word, field in entry:
            if word in index_dict[file_num]:
                wordPostings[word]= (index_dict[file_num][word], field)
            else:
                wordPostings[word]= ('','')
    return wordPostings 

def preprocessFieldQuery(query):
    fieldInfo= {}
    fileWordMap= {}
    
    for f in fields:
        field= query.find(f)
        if field !=-1:
            fieldInfo[field]= f
    
    fieldInfo= sorted(fieldInfo.items())
    fieldInfo.append((1234, ""))            #fake dummy entry
    i=0
    while i+1 <len(fieldInfo):
        field= fieldInfo[i][1].strip(":")
        fieldQuery = (query[fieldInfo[i][0]+2 : fieldInfo[i+1][0]]).lower()
        fieldQuery = preprocess(fieldQuery)

        for word in fieldQuery:
            file_num= getIndexFile(word)
            if file_num not in fileWordMap:
                fileWordMap[file_num] =[(word, field)]
            else:
                fileWordMap[file_num].append((word, field))
        i +=1
    return fileWordMap				#{indexfilenum : (word,field)}

def preprocessPlainQuery(query):
    FileWordMap = dict()
    word = preprocess(query)
    #         print(word)
    for w in word:
        # print(w)
        file_num = getIndexFile(w)
        # print(file_num)
        if file_num != -1:
            if file_num not in FileWordMap:
                FileWordMap[file_num] = [w]
            else:
                FileWordMap[file_num].append(w)
    return FileWordMap
#{4:Sachin, 5:Tendulkar}

def loadInvertedIndex(FileWordMap):
    global index_dict
    for file_num in FileWordMap.keys():
        if file_num not in index_dict.keys():
            fileo = open(inverted_index_path.format(str(file_num)),"rb")
            index_dict[file_num] = pickle.load(fileo)
        else:
            pass

def getTop(tfidfScores):
    global titles_dict
    K = 10
    kRelevant = []
    for key, value in tfidfScores.items():
        if(K ==0):
            break
        x=math.ceil(int(key)/documentLimit)
        kRelevant.append((key, titles_dict[x][int(key)][0]))
        K -=1
    return kRelevant

def writeOutput(kRelevant,time,file):
    of = open(file,"a")
    for k in kRelevant:
        of.write(k[0]+", "+k[1]+"\n")
    of.write(str(time)+"\n\n")
    of.close()

if __name__ == "__main__" :

    TOTALDOCS = loadTitles()
    fileopen = open(queries_path,"r")
    query_list = fileopen.readlines()
    # print(query_list)
    for query in query_list:
        FileWordMap = dict()
        if ":" in query:
            FileWordMap= preprocessFieldQuery(query)
            # print(FileWordMap)
            loadInvertedIndex(FileWordMap)
            ts = time.time()
            postingListsField = getPostingListField(FileWordMap)
            tfidfScores = tfidfField(postingListsField)
            te = time.time()

        else:
            FileWordMap = preprocessPlainQuery(query)
            loadInvertedIndex(FileWordMap)
            ts = time.time()
            postingLists = getPostingList(FileWordMap)
            tfidfScores = tfidfPlain(postingLists)
            te = time.time()

        time1 = te-ts
        tsf = time.time()
        tfidfScores= OrderedDict(sorted(tfidfScores.items(), key=lambda t: t[1], reverse= True))
        topMatch = getTop(tfidfScores)
        tef = time.time()
        time_final = (tef-tsf)+time1
        writeOutput(topMatch,time_final,output_file_path)
        index_dict.clear()
