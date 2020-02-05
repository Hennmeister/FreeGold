import gensim
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import os.path
import numpy as np
import pickle
import spacy

class doc_vec:

    def __init__(self):
        #if os.path.isfile("doc2vec.model"):
        #    self.load_model()
        #else:
        #    self.train_model()
        self.nlp = spacy.load("en_core_web_lg")


    def load_model(self):
        self.model = gensim.models.Doc2Vec.load("doc2vec.model")


    def train_model(self):
        data = pd.read_json("Complied_data.json", orient="split")
        text = [f.strip().split(" ") for f in data["Title"]]

        train_text = [TaggedDocument(s, [i]) for i, s in enumerate(text)]
        self.model = gensim.models.Doc2Vec()
        self.model.build_vocab(train_text)
        self.model.train(train_text, total_words=100000, epochs=1)
        self.model.save("doc2vec.model")

    def evaluate(self, title):
        return np.array(self.nlp(title).title)

    def load_data(self):
        if os.path.isfile("vec_data.json"):
            print("Starting to load data")
            f = open("vec_data.json", "rb")
            self.data = pickle.load(f)
            f.close()
            print("Loaded the data")
        else:
            print("Loading raw data")
            self.data = pd.read_json("Complied_data.json",orient="split")
            print("Getting title")
            self.data = [f.strip() for f in self.data["Title"]]
            self.data = self.data[:50_000]
            #data = [self.model.infer_vector(title) for title in data]
            print("Getting word vectors")
            self.data = [self.nlp(title).vector for title in self.data]
            print("Saving data")
            f = open("vec_data.json", "wb")
            pickle.dump(self.data, f)
            f.close()

    def generate_faiss(self):
        import faiss
        self.dataset = faiss.IndexFlatL2(300)
        self.load_data()
        print("Creating dataset")
        self.dataset.add(np.array(self.data))
        print("Finished creating the dataset")


    def get_closest_faiss(self, word):
        w = self.nlp(word).vector
        print(w)
        return self.dataset.search(np.array(w), 10)



if __name__ == "__main__":
    v = doc_vec()
    v.generate_faiss()
    print(v.evaluate("Hey Test!"))
    print(v.evaluate("Hey Test!").shape)