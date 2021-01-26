import os
import re
import string
'''
Author: Apoorv Sharma
Description:
File to clean tokenize and clean spanish dataset
'''

class SpanishData:
    def __init__(self, spanish_corpus_path) -> None:
        if not os.path.exists(spanish_corpus_path):
            return None
        
        file_list = os.listdir(spanish_corpus_path)
        text_list = []

        for file in file_list:
            file_path = os.path.join(spanish_corpus_path, file)
            with open(file_path, "r", encoding='latin-1') as file:
                text = self._clean_text(file.read())
                text_list.append(text)
        
        spanish_words_char_list = self._flatten(text_list)
        self.spanish_words_char_list = spanish_words_char_list

    def get_spanish_chars(self) -> list:
        return self.spanish_words_char_list
    
    def _clean_text(self, text):
        # get rid of all the XML markup
        text = re.sub('<.*>','',text)

        # get rid of the "ENDOFARTICLE." text
        text = re.sub('ENDOFARTICLE.','',text)

        # get rid of punctuation (except periods)
        punctuationNoPeriod = "[" + re.sub("\.","", string.punctuation) + "]"
        text = re.sub(punctuationNoPeriod, "", text)

        return text
    
    def _flatten(self, l):
        return [word for sublist in l for word in sublist]

if __name__ == '__main__':
    print("Spanish Data Cleaner")
    SpanishData('/job/data/spanish/')