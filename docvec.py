import gensim
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import LoadToMemory
x = open("Shakespeare.txt", "r")
file = x.readlines()
#text = list(file["Title"])
text = [f.strip().split(" ") for f in file]

train_text = [TaggedDocument(s, [i]) for i, s in enumerate(text)]
model = gensim.models.Doc2Vec()
model.build_vocab(train_text)
model.train(train_text, total_words=100000, epochs=1)
while True:
    test = input().split(" ")
    inferred_vector = model.infer_vector(test)
    sims = model.docvecs.most_similar([inferred_vector], topn=10)
    print(sims)
    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document ({}): «{}»\n'.format(1, ' '.join(test)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_text[sims[index][0]].words)))
