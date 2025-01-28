import octis.dataset
import topmost 
import octis
import torch
import numpy as np

class TopMost2OctisAdapter:
    """
    A class that allows to use the TopMost implementation similar to an Octis model
    """

    NoPreprocessing = topmost.preprocessing.Preprocessing(
        keep_num = True,
        keep_alphanum= True,
        min_length= 0,
    )

    def __init__(
            self, 
            model_topmost, 
            model_kwargs: dict = None,
            data2_additional_kwargs: callable = None,
            batch_size: int = 256,
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ):
        """
        Args:
            model_topmost: a TopMost model
            model_kwargs: a dictionary of additional arguments to pass to the model
            data2_additional_kwargs: a function that takes the input data and returns a dictionary of additional arguments
            batch_size: the batch size to use for the model
            device: the device to use for the model
        """

        self.model_topmost = model_topmost
        self.model_kwargs = model_kwargs
        self.data2_additional_kwargs = data2_additional_kwargs
        self.device = device
        self.batch_size = batch_size

    def fit(self,
            dataset: octis.dataset,
            doc_embed_model="all-MiniLM-L6-v2"):
        """
        Fit the model on the given data
        Args:
            dataset: the dataset to fit the model on
            doc_embed_model: the document embedding model to use
        """

        corpus = dataset._corpus 
        vocab = dataset._vocabulary

        # only keep words in the vocabulary
        vocab_set = set(vocab)
        corpus = [[word for word in doc if word in vocab_set] for doc in corpus]

        # join the words back into documents
        corpus = [" ".join(doc) for doc in corpus]


        dataset = topmost.data.RawDataset(docs = corpus, 
                                                   preprocessing = self.NoPreprocessing,
                                                   device=self.device,
                                                   batch_size = self.batch_size,
                                                   doc_embed_model = doc_embed_model)
        print(dataset.vocab_size)

        data_kwargs = self.data2_additional_kwargs(
            corpus = corpus,
            vocab = vocab
        )
        total_kwargs = {**self.model_kwargs, **data_kwargs}
        model = self.model_topmost(
            **total_kwargs
        
        ).to(self.device)
        trainer = topmost.trainers.BasicTrainer(
            model = model,
            dataset=dataset,
        )

        top_words, train_theta = trainer.train()

        # split topwords

        top_words = [[word for word in top.split()] for top in top_words]
        top_words = np.array(top_words)

        beta = trainer.get_beta()

        return top_words, beta

            