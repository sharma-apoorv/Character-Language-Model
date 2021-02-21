import os
import re
import string
import pandas as pd

class LanguageCleaner:
    def __init__(self):
        self.language = None
    
    def _flatten(self, l):
        return [word for sublist in l for word in sublist]

class SpanishCleaner(LanguageCleaner):
    def __init__(self, spanish_corpus_path, save_file_path="", save_sentences=False):
        super().__init__()

        if os.path.isfile(save_file_path):
            self.spanish_sentence_list = self.read_sentence_list_from_file(save_file_path)
            return

        if not os.path.exists(spanish_corpus_path):
            return None
        
        file_list = os.listdir(spanish_corpus_path)
        spanish_sentence_list = []

        for file in file_list:
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
        sentence_list = []

        # open file and read the content in a list
        with open(filePath, 'r', encoding='latin-1') as fh:
            sentence_list = fh.readlines()
        
        sentence_list = map(str.strip, sentence_list)
        return list(sentence_list)

    def clean_text(self, text):
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

class RussianCleaner(LanguageCleaner):
    def __init__(self, russian_corpus_path, save_file_path="", save_sentences=False):
        super().__init__()
        
        if os.path.isfile(save_file_path):
            self.russian_sentence_list = self.read_sentence_list_from_file(save_file_path)
            return

        if not os.path.exists(russian_corpus_path):
            return None
        
        df = pd.read_csv(russian_corpus_path)
        text_list = df['text'].tolist()

        self.russian_sentence_list = text_list

        if save_sentences:
            self.save_sentence_list_to_file(save_file_path)
        

    def get_sentence_list(self) -> list:
        return self.russian_sentence_list
    
    def get_corpus_wowrds(self) -> list:
        res = list(map(str.split, self.russian_sentence_list)) 
        return self._flatten(res)
    
    def save_sentence_list_to_file(self, filePath):
        with open(filePath, 'w', encoding='utf-8') as fh:
            for sentence in self.russian_sentence_list:
                fh.write('%s\n' % sentence)
    
    def read_sentence_list_from_file(self, filePath):
        sentence_list = []

        # open file and read the content in a list
        with open(filePath, 'r', encoding='utf-8') as fh:
            sentence_list = fh.readlines()
        
        sentence_list = map(str.strip, sentence_list)
        return list(sentence_list)

class WortschatzLanguageParser():
    def __init__(self, sentences_file_path, save_file_path="", save_sentences=False) -> None:

        if os.path.isfile(save_file_path):
            self.sentence_list = self.read_sentence_list_from_file(save_file_path)
            return

        if not os.path.exists(sentences_file_path):
            print(f"Error! File Path {sentences_file_path} not found")
            return None
        
        sentences_list = []
        with open(sentences_file_path, "r") as file:
            sentences_list = file.readlines()
        
        cleaned_sentence = []
        for sentence in sentences_list:
            sentence = sentence.split('\t')[1].strip()
            
            if len(sentence) >= 5:
                cleaned_sentence.append(self.clean_sentence(sentence))

        self.sentence_list = cleaned_sentence

        if save_sentences:
            self.save_sentence_list_to_file(save_file_path)

    
    def clean_sentence(self, text):
        # remove numbers
        text_nonum = re.sub(r'\d+', '', text)

        # text_no_punct = text_nonum.translate(str.maketrans('', '', string.punctuation))
        punctuationNoPeriod = "[" + re.sub("\.","", string.punctuation) + "]"
        text_no_punct = re.sub(punctuationNoPeriod, "", text_nonum)

        # substitute multiple whitespace with single whitespace
        # Also, removes leading and trailing whitespaces
        text_no_doublespace = re.sub('\s+', ' ', text_no_punct).strip()
        return text_no_doublespace
    
    def get_sentence_list(self) -> list:
        return self.sentence_list
    
    def save_sentence_list_to_file(self, filePath):
        with open(filePath, 'w') as fh:
            for sentence in self.sentence_list:
                fh.write('%s\n' % sentence)
    
    def read_sentence_list_from_file(self, filePath):
        sentence_list = []

        # open file and read the content in a list
        with open(filePath, 'r') as fh:
            sentence_list = fh.readlines()
        
        sentence_list = map(str.strip, sentence_list)
        return list(sentence_list)
        
if __name__ == '__main__':
    arabic = WortschatzLanguageParser('data/ara_news_2017_1M-sentences.txt', 'data/arabic.txt', True)
    dutch = WortschatzLanguageParser('data/deu_mixed-typical_2011_1M-sentences.txt', 'data/dutch.txt', True)
    french = WortschatzLanguageParser('data/fra_newscrawl-public_2019_1M-sentences.txt', 'data/french.txt', True)
    luxemborgish = WortschatzLanguageParser('data/ltz-lu_web_2013_1M-sentences.txt', 'data/luxemborgish.txt', True)