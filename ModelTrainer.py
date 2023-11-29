
import pandas as pd
import re
from nltk.probability import FreqDist
from nltk.corpus import stopwords 
import spacy
from gensim import corpora, models
import nltk
nltk.download(stopwords)
nltk.download

class ModelTrainer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self._preprocess_data()
        self._train_lda_model()

    def _preprocess_data(self):
        # Explicitly import stopwords from nltk.corpus
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words('english'))
        self.df['Text'] = self.df['Text'].apply(lambda x: re.sub("[^a-zA-Z]", " ", str(x)).lower())
        self.df['OriginalComment'] = self.df['Text']  # Create an 'OriginalComment' column with the original comments
        self.df['Text'] = self.df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    def _train_lda_model(self):
        texts = [text.split() for text in self.df['Text']]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        self.lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=2)

    def get_lda_topics(self, keyword):
        keyword_bow = self.lda_model.id2word.doc2bow(keyword.lower().split())
        lda_topic_distribution = self.lda_model[keyword_bow]
        return lda_topic_distribution
