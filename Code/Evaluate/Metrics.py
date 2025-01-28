from octis.dataset.dataset import Dataset as OCDataset
from octis.evaluation_metrics.metrics import AbstractMetric
from octis.evaluation_metrics.coherence_metrics import Coherence

import re
from tqdm import tqdm

import numpy as np

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

import gensim
gensim_stopwords = gensim.parsing.preprocessing.STOPWORDS

import nltk
nltk.download('brown')
nltk.download('stopwords')
from nltk.corpus import brown as nltk_words
from nltk.corpus import stopwords

from sentence_transformers import SentenceTransformer

from sentence_transformers import SentenceTransformer
#from EvalutateModel import FitEvaluate
from octis.evaluation_metrics.coherence_metrics import Coherence, WECoherencePairwise
#from AdditionalModels.Top2vec_Octis import Top2vec_octis
#from Transfromer2Topic_octis.Transformer2Topic_octis import Transformer2Topic_octis
import nltk
from sklearn.metrics.pairwise import cosine_similarity

from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

nltk_stopwords = stopwords.words('english')

stopwords = list(set(nltk_stopwords + list(gensim_stopwords) + list(ENGLISH_STOP_WORDS)))



class NPMI(AbstractMetric):
    def __init__(self, reference_corpus="20NG", stopwords=stopwords, n_topics=20):
        self.reference_corpus=reference_corpus
        self.stopwords = stopwords
        self.ntopics = n_topics

    def _create_files_20news(self, type="all"):
        dataset_ = OCDataset()
        dataset_.fetch_dataset("20NewsGroup")
        files = dataset_.get_corpus()
        files = [" ".join(words) for words in files]
        return files

    def _create_files_reuters(self, type="all"):
        t = type
        if type == "valid":
            t = "test"

        documents = reuters.fileids()
        id = [d for d in documents if d.startswith(t)]
        files = np.array([reuters.raw(doc_id) for doc_id in id])

        return files

    def create_files_BBCNews(type):
        dataset_ = OCDataset()
        dataset_.fetch_dataset("BBC_News")
        files = dataset_.get_corpus()
        files = [" ".join(words) for words in files]
        return files

    def create_files_M10(type):
        dataset_ = OCDataset()
        dataset_.fetch_dataset("M10")
        files = dataset_.get_corpus()
        files = [" ".join(words) for words in files]
        return files

    def _create_vocab_preprocess(self, data, preprocess, process_data=False):
        word_to_file = {}
        word_to_file_mult = {}


        process_files = []
        for file_num in range(0, len(data)):
            words = data[file_num].lower()
            words = words.strip()
            words = re.sub('[^a-zA-Z0-9]+\s*', ' ', words)
            words = re.sub(' +', ' ', words)
            #.translate(strip_punct).translate(strip_digit)
            words = words.split()
            #words = [w.strip() for w in words]
            proc_file = []

            for word in words:
                if word in self.stopwords  or word =="dlrs" or word == "revs":
                    continue
                if word in word_to_file:
                    word_to_file[word].add(file_num)
                    word_to_file_mult[word].append(file_num)
                else:
                    word_to_file[word]= set()
                    word_to_file_mult[word] = []

                    word_to_file[word].add(file_num)
                    word_to_file_mult[word].append(file_num)

            process_files.append(proc_file)

        for word in list(word_to_file):
            if len(word_to_file[word]) <= preprocess  or len(word) <= 3:
                word_to_file.pop(word, None)
                word_to_file_mult.pop(word, None)

        if process_data:
            vocab = word_to_file.keys()
            files = []
            for proc_file in process_files:
                fil = []
                for w in proc_file:
                    if w in vocab:
                        fil.append(w)
                files.append(" ".join(fil))

            data = files

        return word_to_file, word_to_file_mult, data



    def _create_vocab_and_files(self, dataset, type, preprocess=5):
        data = None
        if dataset == "20NG":
            data = self._create_files_20news(type)
        elif dataset == "reuters":
            data = self._create_files_reuters(type)

        return self._create_vocab_preprocess(data, preprocess)


    def score(self, model_output):

        topic_words = model_output['topics']
        word_doc_counts, dev_word_to_file_mult, dev_files = self._create_vocab_and_files(dataset=self.reference_corpus, preprocess=1, type="all")
        nfiles = len(dev_files)
        eps = 10**(-12)

        all_topics = []
        for k in range(self.ntopics):
            topic_score = []

            ntopw = len(topic_words[k])

            for i in range(ntopw-1):
                for j in range(i+1, ntopw):
                    w1 = topic_words[k][i]
                    w2 = topic_words[k][j]

                    w1w2_dc = len(word_doc_counts.get(w1, set()) & word_doc_counts.get(w2, set()))
                    w1_dc = len(word_doc_counts.get(w1, set()))
                    w2_dc = len(word_doc_counts.get(w2, set()))

                    # Correct eps:
                    pmi_w1w2 = np.log((w1w2_dc * nfiles) / ((w1_dc * w2_dc) + eps) + eps)
                    npmi_w1w2 = pmi_w1w2 / (- np.log( (w1w2_dc)/nfiles + eps))

                    topic_score.append(npmi_w1w2)

            all_topics.append(np.mean(topic_score))

        avg_score = np.around(np.mean(all_topics), 5)


        return avg_score


    def npmi_per_topic(self, topic_words, ntopics, preprocess=5, type="all"):

        word_doc_counts, dev_word_to_file_mult, dev_files = self._create_vocab_and_files(dataset=self.reference_corpus, preprocess=preprocess, type=type)
        nfiles = len(dev_files)
        eps = 10**(-12)

        all_topics = []
        for k in range(ntopics):
            topic_score = []

            ntopw = len(topic_words[k])

            for i in range(ntopw-1):
                for j in range(i+1, ntopw):
                    w1 = topic_words[k][i]
                    w2 = topic_words[k][j]

                    w1w2_dc = len(word_doc_counts.get(w1, set()) & word_doc_counts.get(w2, set()))
                    w1_dc = len(word_doc_counts.get(w1, set()))
                    w2_dc = len(word_doc_counts.get(w2, set()))

                    # Correct eps:
                    pmi_w1w2 = np.log((w1w2_dc * nfiles) / ((w1_dc * w2_dc) + eps) + eps)
                    npmi_w1w2 = pmi_w1w2 / (- np.log( (w1w2_dc)/nfiles + eps))

                    topic_score.append(npmi_w1w2)

            all_topics.append(np.mean(topic_score))

        results = {}
        for k in range(ntopics):
            results[", ".join(topic_words[k])] = np.around(all_topics[k],5)


        return results
    

def Embed_corpus(dataset, embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")):
    """
    Create a dictionary with the word embedding of every word in the dataset.
    Use the embedder.
    If the file 'Embeddings/{emb_filename}.pickle' is available, read the embeddings from this file.

    Otherwise create new embeddings.
    Returns the embedding dict
    """


    emb_dic = {}
    word_list = []
    for doc in dataset.get_corpus():
        for word in doc:
            word_list.append(word)
    word_list = set(word_list)
    for word in tqdm(word_list):
        emb_dic[word] = embedder.encode(word)

    return emb_dic

def Update_corpus_dic_list(corpus, emb_dic, embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")):
    """
    Updates embedding dict with embeddings in word_lis

    """
    if corpus == "brown":
      word_lis = nltk_words.words()
      word_lis = [word.lower().strip() for word in word_lis]
      word_lis = [re.sub('[^a-zA-Z0-9]+\s*', '', word) for word in word_lis]
      word_lis = list(set(word_lis))

    keys = set(emb_dic.keys())
    for word in tqdm(set(word_lis)):
        if word not in keys:
            emb_dic[word] = embedder.encode(word)

    return emb_dic

def Embed_topic(topics_tw, corpus_dict,  n_words = 10, embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")):
    """
    takes the list of topics and embed the top n_words words with the corpus dict
    if possible, else use the embedder.
    """
    topic_embeddings = []
    for topic in tqdm(topics_tw):
        if n_words != None:
            topic = topic[:n_words]

        add_lis = []
        for word in topic:
            try:
                add_lis.append(corpus_dict[word])
            except KeyError:
                #print(f'did not find key {word} to embedd topic, create new embedding...')
                add_lis.append(embedder.encode(word))

        topic_embeddings.append(add_lis)

    return topic_embeddings

def Embed_stopwords(stopwords, embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")):
    """
    take the list of stopwords and embeds them with embedder
    """

    sw_dic = {}  #first create dictionary with embedding of every unique word
    stopwords_set = set(stopwords)
    for word in tqdm(stopwords_set):
        sw_dic[word] = embedder.encode(word)

    sw_list =[]
    for word in stopwords:      #use this dictionary to embed all the possible stopwords
        sw_list.append(sw_dic[word])

    return sw_list

def mean_over_diag(mat):
    """
    Calculate the average of all elements of a quadratic matrix
    that are above the diagonal
    """
    h, w = mat.shape
    assert h==w, 'matrix must be quadratic'
    mask = np.triu_indices(h, k = 1)
    return np.mean(mat[mask])

def cos_sim_pw(mat):
    """
    calculate the average cosine similarity of all rows in the matrix (but exclude the similarity of a row to itself)
    """
    sim = cosine_similarity(mat)
    return mean_over_diag(sim)

class Embedding_Coherence(AbstractMetric):
    """
    Average cosine similarity between all top words in a topic
    """

    def __init__(self, corpus_dict, n_words = 10):
        """
        corpus_dict: dict that maps each word in the corpus to its embedding
        n_words: number of top words to consider
        """

        self.n_words = n_words
        self.corpus_dict = corpus_dict



    def score_per_topic(self, model_output):

        topics_tw = model_output['topics']


        emb_tw = Embed_topic(topics_tw, self.corpus_dict,  self.n_words)  #embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2,0,1)[:, :self.n_words, :]  #create tensor of size (n_topics, n_topwords, n_embedding_dims)
        self.embeddings = emb_tw



        topic_sims = []
        for topic_emb in emb_tw:                          #for each topic append the average pairwise cosine similarity within its words
            topic_sims.append(cos_sim_pw(topic_emb))

        return np.array(topic_sims)


    def score(self, model_output):
        res = self.score_per_topic(model_output)
        return sum(res)/len(res)

class Embedding_Topic_Diversity(AbstractMetric):
    """
    Measure the diversity of the topics by calculating the mean cosine similarity
    of the mean vectors of the top words of all topics
    """

    def __init__(self, corpus_dict, n_words = 10):
        """
        corpus_dict: dict that maps each word in the corpus to its embedding
        n_words: number of top words to consider
        """

        self.n_words = n_words
        self.corpus_dict = corpus_dict



    def score(self, model_output):

        topics_tw = model_output['topics']  #size: (n_topics, voc_size)
        topic_weights = model_output['topic-word-matrix'][:, :self.n_words]  #select the weights of the top words

        topic_weights = topic_weights/np.nansum(topic_weights, axis = 1).reshape(-1, 1) #normalize the weights such that they sum up to one


        emb_tw = Embed_topic(topics_tw, self.corpus_dict,  self.n_words)  #embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2,0,1)[:, :self.n_words, :]  #create tensor of size (n_topics, n_topwords, n_embedding_dims)



        weighted_vecs = topic_weights[:, :, None] * emb_tw  #multiply each embedding vector with its corresponding weight
        topic_means = np.nansum(weighted_vecs, axis = 1) #calculate the sum, which yields the weighted average

        return float(cos_sim_pw(topic_means))


    def score_per_topic(self, model_output):

        topics_tw = model_output['topics']  #size: (n_topics, voc_size)
        topic_weights = model_output['topic-word-matrix'][:, :self.n_words]  #select the weights of the top words size: (n_topics, n_topwords)

        topic_weights = topic_weights/np.nansum(topic_weights, axis = 1).reshape(-1, 1) #normalize the weights such that they sum up to one


        emb_tw = Embed_topic(topics_tw, self.corpus_dict,  self.n_words)  #embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2,0,1)[:, :self.n_words, :]  #create tensor of size (n_topics, n_topwords, n_embedding_dims)
        self.embeddings = emb_tw


        weighted_vecs = topic_weights[:, :, None] * emb_tw  #multiply each embedding vector with its corresponding weight
        topic_means = np.nansum(weighted_vecs, axis = 1) #calculate the sum, which yields the weighted average

        sim = cosine_similarity(topic_means)   #calculate the pairwise cosine similarity of the topic means
        sim_mean = (np.nansum(sim, axis = 1) - 1)/(len(sim)-1)  #average the similarity of each topic's mean to the mean of every other topic

        return sim_mean

class Perplexity(AbstractMetric):
    """
    Implement perplxity metric
    """
    def __init__(self, base: float = np.exp(1)):
        self.base = base

    def score(self, nll_mean: float) -> float:
        """
        Compute perplexity from the mean negative log likelihood
        Args:
        nll_lis: list of negative log likelihoods
        """
        return nll_mean


def get_tw_embeddings(dataset):
    """
    Get embeddings for a dataset
    Params:
    dataset: OCDataset object
    Returns:
    tw_emb: dict with embeddings for all words in the dataset
    """

    metric_embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2") #model to compute the embedding metrics with

    tw_emb = Embed_corpus(dataset, metric_embedder)
    tw_emb = Update_corpus_dic_list("brown", tw_emb, metric_embedder)

    return tw_emb


def score_all(dataset, tw_emb, n_words, result, validation_loss_mean = None):
    """
    Compute all metrics for a dataset
    Params:
    dataset: OCDataset object
    tw_emb: dict with embeddings for all words in the dataset
    n_words: number of top words to consider
    result: dict with model output
    validation_loss_mean: list of validation losses to compute perplexity
    """

    metrics = [
    Coherence(texts = dataset.get_corpus()),
    WECoherencePairwise(),
    Embedding_Coherence(tw_emb, n_words=n_words),
    Embedding_Topic_Diversity(tw_emb, n_words=n_words),
    TopicDiversity(topk=10)
    ]

    metrics_names = ['NPMI', 'WE_CO_PW', 'Embedding_Coherence', "WESS", 'Topic Diversity']

    metrics_dic = {name:metric for name, metric in zip(metrics_names, metrics)}

    score_dic = {}
    for name, metric in tqdm(metrics_dic.items()):
        score_dic[name] = metric.score(result)

    if validation_loss_mean is not None:
        score_dic['Perplexity'] = Perplexity().score(validation_loss_mean)

    return score_dic

