import gensim
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import os.path
import numpy as np
import pickle

class doc_vec:

    def __init__(self):
        if os.path.isfile("doc2vec.model"):
            self.load_model()
        else:
            self.train_model()


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
        return np.array(self.model.infer_vector([title]))

    def generate_faiss(self):
        import faiss
        self.dataset = faiss.IndexFlatL2(100)
        if os.path.isfile("vec_data.json"):
            f = open("vec_data.json", "r")
            data = pickle.load(f)
            f.close()
        else:
            data = pd.read_json("Complied_data.json")
            data = [f.strip().split(" ") for f in data["Title"]]
            data = [self.model.infer_vector(title) for title in data]
            f = open("vec_data.json", "w")
            pickle.dump(data, f)
            f.close()
        for d in data:
            self.dataset.add(d)




if __name__ == "__main__":
    v = doc_vec()
    print(v.evaluate("Test"))
    print(v.evaluate("Test").shape)
    # while True:
    #     test = input().split(" ")
    #     inferred_vector = model.infer_vector(test)
    #     sims = model.docvecs.most_similar([inferred_vector], topn=10)
    #     print(sims)
    #     # Compare and print the most/median/least similar documents from the train corpus
    #     print('Test Document ({}): «{}»\n'.format(1, ' '.join(test)))
    #     print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    #     for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    #         print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_text[sims[index][0]].words)))
