import pickle
from Code.Evaluate.Metrics import score_all, get_tw_embeddings
from Benchmark.OctisDataset2StreamDataset import OctisDataset2StreamDataset
import torch
from tqdm import tqdm

import sys 

sys.path.append("../")


class Benchmark:

    def __init__(
            self,
            octis_dataset,
            embedding_df,
            models: list,
            model_specific_data2params_fun_list: list,
            n_topics:list = [20, 200],
            batch_size: int = 256,
            embeddings_path = "../",
            embeddings_file_path = "../"
    ):
        """"
        Args:
            octis_dataset
            embedding_df: a dataframe containing the embeddings
            models: a list of models to benchmark
            model_specific_data2params_fun_list: a list of functions that take the input data and return a dictionary of additional arguments
            n_topics: a list of numbers of topics to use
            batch_size: the batch size to use for the model	
            embeddings_path: the path to the embeddings
            embeddings_file_path: the path to the embeddings file
        """
        octis_dataset = OctisDataset2StreamDataset(octis_dataset)

        self.octis_dataset = octis_dataset
        self.embedding_df = embedding_df
        self.models = models
        self.model_specific_data2params_fun_list = model_specific_data2params_fun_list
        self.n_topics = n_topics
        self.batch_size = batch_size

        self.embeddings_path = embeddings_path
        self.embeddings_file_path = embeddings_file_path


    def setup(self):

        

        octis_dataset = self.octis_dataset
        embedding_df = self.embedding_df

        corpus = self.octis_dataset.get_corpus()
        tw_emb = get_tw_embeddings(octis_dataset)
        
        embedding_df.sort_values(by = "word", inplace = True)
        embedding_ten_lis = []

        for i in range(len(embedding_df)):
            embedding_ten_lis.append(embedding_df["embedding"].iloc[i])
        embedding_df.sort_values(by = "word", inplace = True)
        embedding_ten_lis = []

        embedded_words = embedding_df.index.tolist()

        for i in range(len(embedding_df)):
            embedding_ten_lis.append(embedding_df["embedding"].iloc[i])
        embedding_ten = torch.stack(embedding_ten_lis)  


        self.ocits_dataset = octis_dataset
        self.tw_emb = tw_emb
        self.embedding_df = embedding_df
        self.embedding_ten = embedding_ten
        self.embedded_words = embedded_words
        self.corpus = corpus

        self.ocits_dataset.name = "dataset"
        
        self.ocits_dataset.get_labels = lambda: [1 for _ in range(len(self.ocits_dataset.get_corpus()))]

    def benchmark_model(self, 
                        model, 
                        #data2params_fun, 
                        n_topics):
        """
        benchmark a model
        """

        mod = model(n_topics)

        mod.embeddings_path =  self.embeddings_path
        mod.embeddings_file_path = self.embeddings_file_path

        res = mod.train_model(self.ocits_dataset)

        res
        
        evaluation_result = score_all(
            dataset = self.octis_dataset,
            tw_emb=self.tw_emb,
            n_words=10,
            result = {'topics': res["topics"], 
                    "topic-word-matrix": res["topic-word-matrix"]},
        )

        return evaluation_result
    
    def run(self):
        """
        Run the benchmark
        """

        self.setup()

        results = {}

        assert len(self.models) == len(self.model_specific_data2params_fun_list), "The number of models and the number of data2params functions must be the same"
        for model, data2params_fun in tqdm(list(zip(self.models, self.model_specific_data2params_fun_list))):
            for n_topics in self.n_topics:
                try:
                    r = self.benchmark_model(
                        model = model,
                        n_topics = n_topics
                    )
                except Exception as e:
                    # if keyboard interrupt, actually interrupt
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    else:
                        r = e

                results[(model, n_topics)] = r

                print(f"Done with model {model} and n_topics {n_topics}")

        return results