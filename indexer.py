import re
import os
import sys
import time
import math
import pickle
import xml.sax
from heapq import heapify, heappush, heappop 
from Stemmer import Stemmer
from nltk.corpus import stopwords
from collections import defaultdict

total_tokens,doc_id, index_words, index_file_cnt, titleFileCount = 0, 1, 0, 1, 1
global_dict,page_dict, final_dict = dict(),dict(), dict()
range_dict = defaultdict(dict)
titles = dict()
documentLimit = 50000
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

stemmer = Stemmer("english")
stop_words = set(stopwords.words('english'))

infoBoxRegex = "{{Infobox((.|\n)*?)}}"
categoryRegex = "\[\[Category:(.*?)\]\]"
referenceRegex = "== *[rR]eferences *==((.*?\n)*?)\n"
externalLinksRegex = "== *[eE]xternal [lL]inks *==\n((.*?\n)*?)\n"
externalLinksRegex = "== *[eE]xternal [lL]inks *==\n(((.*?\n)*?)\n) *?\{+?"

def cleanText(data):
    data = re.sub(r'{\|(.*?)\|}','',data, flags = re.DOTALL)
    data = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', data, flags=re.DOTALL) 
    data = re.sub(r'<(.*?)>', '', data)
    return data

def tokenization(content):
    global total_tokens
    tokens = re.findall("\d+|[\w]+",content)
    tokens = [t for t in tokens]
    total_tokens += len(tokens)
    return tokens

def preprocess(data):
    return [stemmer.stemWord(token.casefold()) for token in tokenization(data) if token and token.casefold() not in stop_words]

def createSectionDict(text):
    data = str(text)
    data = cleanText(data)
    tokens = preprocess(data)
    temp = {}
    i=0
    for k in tokens:
        if len(k)<=2:
            continue
        elif k.isdecimal() and len(k)>4:  
            continue
        elif k in temp:
            temp[k] += 1
        else:
            temp[k]=1
    
    return temp

def createPageDict(input_dict, index):
    global global_dict
    global page_dict
    for key in input_dict.keys():
        if key not in page_dict:
            page_dict[key]= [0,0,0,0,0,0]
        page_dict[key][index] += input_dict[key]

def createGlobalDict(doc_id):  
    for word in page_dict:
        lt = page_dict[word]
        tmpdict = {}
        if word in global_dict:
            tmpdict = global_dict[word]
        tmpdict[doc_id] = lt
        global_dict[word] = tmpdict

def writeToFile(filename):
    global global_dict					#format: [term: d12#t2+i3+b1|d13#t1+r1]
    global index_file_cnt
    inverted_index_file_path = filename.format(str(index_file_cnt))
#     print(inverted_index_file_path)
    index_file_cnt += 1
#     print(filename)
    fields_list = ["t","i","b","c","l","r"]
    with open(inverted_index_file_path,"a") as of:
        for word,docs in sorted(global_dict.items()):
            if not(re.match('^[a-zA-Z0-9]+$',word)) or re.match('^[0]+$',word):
                continue
            docdict = docs
            docfreq = []
            for k,v in sorted(docdict.items()):
                docdet = "d"+str(k)+"#"
                for i in range(6):
                    if v[i]>0:
                        docdet += fields_list[i]+str(v[i])+"+"
                docdet = docdet[:-1]                             # remove last +
                docfreq.append(docdet)
            docfreqstr = '|'.join(docfreq)
            of.write(word+":"+docfreqstr+"\n")  
    of.close()

def preprocessTitle(title):
    title = cleanText(str(title))
    title_tokens = preprocess(title)   
    title_dict={}
    i=0
    for t in title_tokens:
        if len(t)<=2:
            continue
        elif t.isdecimal() and len(t)>4:
            continue
        elif t not in title_dict:
            title_dict[t] = 1
        else:
            title_dict[t] += 1
        
    return title_dict

def writeTitles():
    global titleFileCount
    titles_file= open(titles_FilePath.format(str(titleFileCount)), "wb")
    pickle.dump(titles, titles_file)
    titleFileCount+=1

class WikiHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.current_tag = ""
        self.title = ""
        self.text = ""
        self.tokens = 0
          
    def startElement(self,tag,attr):
        self.current_tag = tag
        if self.current_tag == "page":
            self.title = ""
            self.text = ""
    
    def characters(self, data):
        if self.current_tag == "title":
            self.title += data
        elif self.current_tag == "text":
            self.text += data

    def endElement(self,tag):
        global total_tokens
        global doc_id
        global global_dict
        global page_dict
        global index_words
        global index_file_cnt
        global titles

        if tag == "page":
            doc_id += 1 ## 1: (abc,12)
            titles[doc_id-1] = (self.title,self.tokens)
            page_dict.clear()
            if doc_id % documentLimit == 0:
                if global_dict:
                    #print(global_dict)
                    writeToFile(path_to_inverted_index)
                    writeTitles()
                    global_dict.clear()
                    titles.clear()
            self.tokens = 0

        if tag == "title":
            title_dict = preprocessTitle(self.title)
            createPageDict(title_dict, 0)
            self.tokens +=len(title_dict)

        if tag == "text":
            text = self.text
            
            links = re.findall(externalLinksRegex,text)
            category = re.findall(categoryRegex,text)
            references = re.findall(referenceRegex,text)
            info = re.findall(infoBoxRegex,text)

            contentNoLinks = re.sub(externalLinksRegex,'',text)
            contentNoCategory = re.sub(categoryRegex,'',contentNoLinks)
            contentNoReferences = re.sub(referenceRegex,'',contentNoCategory)
            contentNoInfobox = re.sub(infoBoxRegex,'',contentNoReferences)
            body = re.sub(r'\{\{.*\}\}','', contentNoInfobox)

            info_dict,body_dict,category_dict,references_dict,links_dict = dict(),dict(),dict(),dict(),dict()

            if (len(info)>0):
                info_dict = createSectionDict(info)
            if (len(category)>0):
                category_dict= createSectionDict(category)
            if (len(links)>0):
                links_dict = createSectionDict(links)
            if (len(references)>0):
                references_dict = createSectionDict(references)
            if (len(body)>0):
                body_dict= createSectionDict(body)
                
            self.tokens += (len(info_dict)+len(body_dict)+len(links_dict)+len(references_dict)+len(category_dict))

            createPageDict(info_dict, 1)
            createPageDict(body_dict, 2)
            createPageDict(category_dict, 3)
            createPageDict(links_dict, 4)
            createPageDict(references_dict, 5)
            
            createGlobalDict(doc_id)
            
        self.current_tag =""
        
def write_PrimaryIndices():
    global final_dict
    global range_dict
    
    sorted_IndexDict= dict(sorted(final_dict.items(), key=lambda t: t[0]))
    
    for k,v in sorted_IndexDict.items():

        if k.startswith(check_dict['1']):
            range_dict[1][k] = v

        elif k.startswith(check_dict['2']):
            range_dict[2][k] = v

        elif k.startswith(check_dict['3']):
            range_dict[3][k] = v

        elif k.startswith(check_dict['4']):
            range_dict[4][k] = v

        elif k.startswith(check_dict['5']):
            range_dict[5][k] = v

        elif k.startswith(check_dict['6']):
            range_dict[6][k] = v

        elif k.startswith(check_dict['7']):
            range_dict[7][k] = v

        elif k.startswith(check_dict['8']):
            range_dict[8][k] = v

        elif k.startswith(check_dict['9']):
            range_dict[9][k] = v

        elif k.startswith(check_dict['10']):
            range_dict[10][k] = v

        else:
            pass    
        
    for final_indexNum in range(1,11):
        output_file= open(final_indexes.format(str(final_indexNum)), "wb")
        pickle.dump(range_dict[final_indexNum], output_file) 
        output_file.close()

    final_dict.clear()

def kWayMerge():
    global indexFileCount
    global final_dict
    input_files= []

    for file_num in range (1,indexFileCount+1):
        input_files.append(open(path_to_inverted_index.format(str(file_num)), "r"))
    
    heap = [] 
    heapify(heap)  
    
    for file_num in range (1, indexFileCount+1):
        try:
            line= input_files[file_num-1].readline()
            word= line.split(":")[0]
            value= line.split(":")[1].split("\n")[0]
            heappush(heap, (word, value, file_num))
        except:
            os.remove(path_to_inverted_index.format(str(file_num)))

    while heap:
        entry= heappop(heap)
        word, line, file_num = entry[0], entry[1], entry[2]
        
        
        while heap and heap[0][0] == word:
            line += "|"+ heap[0][1]
            try:
                new_line= input_files[heap[0][2]-1].readline()
                new_word= new_line.split(":")[0]
                new_value= new_line.split(":")[1].split("\n")[0]
                heappush(heap, (new_word, new_value, heap[0][2]))
            except:
                os.remove(path_to_inverted_index.format(str(heap[0][2])))
            heappop(heap)

        final_dict[word]= line
        
        try:
            next_line= input_files[file_num-1].readline()
            next_word= next_line.split(":")[0]
            next_value= next_line.split(":")[1].split("\n")[0]
            heappush(heap, (next_word, next_value, file_num))
        except:
            os.remove(path_to_inverted_index.format(str(file_num)))
        
        if(len(final_dict) == documentLimit or len(heap) ==0):
            write_PrimaryIndices()

def writeToStats():
    token_count =0
    for i in range(1,11):
        fileread = open(final_indexes.format(str(i)),"rb")
        new_dict = pickle.load(fileread)
        token_count+=len(new_dict)
    size = 0
    indexpath = "./indexes"
    for path, dirs, files in os.walk(indexpath):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)

    out = open(inverted_index_stat,"w")
    size = "Inverted Index size: "+str(size/math.pow(10,9))+" GB\n"
    noOfFiles = "No of index files: "+ str(10)+"\n"
    noOfTokens = "No of tokens in inverted index: "+str(token_count)
    out.write(size)
    out.write(noOfFiles)
    out.write(noOfTokens)
    out.close()

def indexing(path_to_xml_dump):
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces,0)
    handler_m = WikiHandler()
    parser.setContentHandler(handler_m)
    parser.parse(path_to_xml_dump)

if __name__ == "__main__":
    path_to_xml_dump = "../..//data"
    path_to_inverted_index = "./indexes/index_{}"
    final_indexes = "./indexes/final_{}"
    inverted_index_stat = "./stats.txt"
    titles_FilePath = "./titles/title_{}"

    indexing(path_to_xml_dump)
    if bool(global_dict):
        writeToFile(path_to_inverted_index)
        global_dict.clear()
    if bool(titles):
        writeTitles()
        titles.clear()
    indexFileCount = math.ceil(doc_id/documentLimit)
    kWayMerge()
    writeToStats()
