import pickle
try:
    from Code.Evaluate.Metrics import score_all, get_tw_embeddings
except:
    from TNTM.Code.Evaluate.Metrics import score_all, get_tw_embeddings
import torch
from tqdm import tqdm

import sys 

sys.path.append("../")

from TNTM.Benchmark.TopMost2OctisAdapter import TopMost2OctisAdapter

class Benchmark:

    def __init__(
            self,
            octis_dataset,
            embedding_df,
            models: list,
            model_specific_data2params_fun_list: list,
            n_topics:list = [20, 200],
            batch_size: int = 256,
    ):
        """"
        Args:
            octis_dataset
            embedding_df: a dataframe containing the embeddings
            models: a list of models to benchmark
            model_specific_data2params_fun_list: a list of functions that take the input data and return a dictionary of additional arguments
            n_topics: a list of numbers of topics to use
            batch_size: the batch size to use for the model	
        """

        self.octis_dataset = octis_dataset
        self.embedding_df = embedding_df
        self.models = models
        self.model_specific_data2params_fun_list = model_specific_data2params_fun_list
        self.n_topics = n_topics
        self.batch_size = batch_size


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

    def benchmark_model(self, 
                        model, 
                        data2params_fun, 
                        n_topics):
        """
        benchmark a model
        """

        model = TopMost2OctisAdapter(
            model_topmost = model,
            model_kwargs= {
                "num_topics": n_topics,
            },
            data2_additional_kwargs = data2params_fun,
            batch_size = self.batch_size
        )

        res = model.fit(self.ocits_dataset)

        topics = res[0]


        evaluation_result = score_all(
            dataset = self.octis_dataset,
            tw_emb=self.tw_emb,
            n_words=10,
            result = {'topics': topics, 
                    "topic-word-matrix": res[1]},
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
                        data2params_fun = data2params_fun,
                        n_topics = n_topics
                    )
                except Exception as e:
                    # dont't do anything with a keyboard interrupt
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    else:
                        r[(model, n_topics)] = None

                results[(model, n_topics)] = r

                print(f"Done with model {model} and n_topics {n_topics}")

        return results