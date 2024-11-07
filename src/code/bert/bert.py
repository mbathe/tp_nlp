
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from bert_score import BERTScorer
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
print(torch.cuda.is_available())


class Bert:
    def __init__(self, model="bert-base-uncased", tokenizer="bert-base-uncased", documents=[]):
        self.model = BertModel.from_pretrained(model)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.sentence_embeddings = None
        self.sentences = documents

    def processing(self):
        """
        Process the input sentences to generate their corresponding sentence embeddings using BERT.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Side Effects:
        ------------
        Updates the 'sentence_embeddings' attribute of the Bert instance with the generated sentence embeddings.
        """
        sentence_embeddings = []
        for text in tqdm(self.sentences):
            inputs = self.tokenizer(text, return_tensors='pt',
                                    max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state
                embedding = outputs.last_hidden_state.mean(dim=1)
                sentence_embeddings.append(embedding.cpu().numpy())
        self.sentence_embeddings = np.array(
            [s[0] for s in sentence_embeddings])
       # print(self.sentence_embeddings.shape)

    def get_resume(self, top_n=1):
        """
        Generate a summary of the document by selecting the top-n most similar sentences.

        Parameters:
        -----------
        top_n : int, optional
            The number of sentences to include in the summary. Default is 1.

        Returns:
        --------
        top_indices : numpy.ndarray
            An array of indices representing the selected sentences in the original document.
        summary : list
            A list of the selected sentences, in the order of their similarity to the document.

        Side Effects:
        ------------
        None
        """
        document_embedding = torch.mean(torch.tensor(
            self.sentence_embeddings), dim=0).unsqueeze(0)
        similarities = cosine_similarity(
            document_embedding, self.sentence_embeddings).flatten()

        # Select the most similar sentences
        top_indices = similarities.argsort()[-top_n:][::-1]
        summary = [self.sentences[i] for i in top_indices]
        return top_indices, summary

    def get_bert_score(self, reference, candidate):
        """
        Calculate the BERT score between the resume document and the original document.

        Parameters:
        -----------
        resume_doc : str
            The resume document to compare.
        doc : str
            The original document to compare.

        Returns:
        --------
        bert_score : float
            The BERT score between the resume document and the original document.

        Side Effects:
        ------------
        None
        """
        scorer = BERTScorer(model_type='bert-base-uncased')
        P, R, F1 = scorer.score([candidate], [reference])
        return P, R, F1
