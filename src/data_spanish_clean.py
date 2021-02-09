import os
import re
import string

'''
Author: Apoorv Sharma
Description:
File to clean tokenize and clean spanish dataset
'''

class SpanishData:
    def __init__(self, spanish_corpus_path, save_file_path="", save_sentences=False) -> None:

        if os.path.isfile(save_file_path):
            self.read_sentence_list_from_file(save_file_path)
            return

        if not os.path.exists(spanish_corpus_path):
            return None
        
        file_list = os.listdir(spanish_corpus_path)
        spanish_sentence_list = []

        for file in file_list[:2]:
            file_path = os.path.join(spanish_corpus_path, file)
            with open(file_path, "r", encoding='latin-1') as file:
                spanish_sentences = self._clean_text(file.read())
                spanish_sentence_list.append(spanish_sentences)
        
        spanish_sentence_list = self._flatten(spanish_sentence_list)
        self.spanish_sentence_list = spanish_sentence_list

        if save_sentences:
            self.save_sentence_list_to_file(save_file_path)

    def get_sentence_list(self) -> list:
        return self.spanish_sentence_list
    
    def get_corpus_wowrds(self) -> list:
        res = list(map(str.split, self.spanish_sentence_list)) 
        return self._flatten(res)
    
    def save_sentence_list_to_file(self, filePath):
        with open(filePath, 'w', encoding='latin-1') as fh:
            for sentence in self.spanish_sentence_list:
                fh.write('%s\n' % sentence)
    
    def read_sentence_list_from_file(self, filePath):
        spanish_sentence_list = []

        # open file and read the content in a list
        with open(filePath, 'r', encoding='latin-1') as fh:
            spanish_sentence_list = fh.readlines()
        
        print(spanish_sentence_list[:10])

    def _clean_text(self, text):
        # get rid of all the XML markup
        text = re.sub('<.*>','',text)

        # get rid of the "ENDOFARTICLE." text
        text = re.sub('ENDOFARTICLE.','',text)

        # get rid of punctuation (except periods)
        punctuationNoPeriod = "[" + re.sub("\.","", string.punctuation) + "]"
        text = re.sub(punctuationNoPeriod, "", text)

        text = text.split("\n")
        text = [i for i in text if i] 
        return text
    
    def _flatten(self, l):
        return [word for sublist in l for word in sublist]

if __name__ == '__main__':
    print("Spanish Data Cleaner")
    sd = SpanishData('./data/spanish/', save_file_path="./data/spanish.txt", save_sentences=True)