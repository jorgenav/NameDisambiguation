import re, pprint, os, numpy
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from nltk.cluster import GAAClusterer
from sklearn.metrics.cluster import adjusted_rand_score



def read_file(file):
    myfile = open(file,"r")
    data = ""
    lines = myfile.readlines()
    for line in lines:
        data = data + line
    myfile.close
    return data

def cluster_texts(texts, clustersNumber, distanceFunction):
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Created a collection of", len(collection), "terms.")

    #get a list of unique terms
    unique_terms = list(set(collection))
    print("Unique terms found: ", len(unique_terms))

    ### And here we actually call the function and create our array of vectors.
    vectors = [numpy.array(TF(f, unique_terms, collection)) for f in texts]
    print("Vectors created.")

    # initialize the clusterer
    # clusterer = GAAClusterer(clustersNumber)
    # clusters = clusterer.cluster(vectors, True)
    clusterer = AgglomerativeClustering(n_clusters=clustersNumber, linkage="average", affinity=distanceFunction)
    clusters = clusterer.fit_predict(vectors)

    return clusters

# Function to create a TF vector for one document. For each of
# our unique words, we have a feature which is the tf for that word
# in the current document
def TF(document, unique_terms, collection):
    word_tf = []
    for word in unique_terms:
        # word_tf.append(collection.tf(word, document))
        # word_tf.append(collection.idf(word))
        word_tf.append(collection.tf_idf(word, document))
    return word_tf

if __name__ == "__main__":
    folder = "../data"
    # Empty list to hold text documents.
    texts = []
    stop = set(stopwords.words('english'))

    listing = os.listdir(folder)
    for file in listing:
        if file.endswith(".txt"):
            url = folder+"/"+file
            f = open(url,encoding="latin-1");
            raw = f.read()
            f.close()
            tokens = nltk.word_tokenize(raw)

########################################################################################################################

            # # Stop words
            # tokens_nostop = []
            # for word in tokens:
            #     if word.lower() not in stop:
            #         tokens_nostop.append(word)
            # # text = nltk.Text(tokens_nostop)

########################################################################################################################

            # # Steamming
            # stemmer = PorterStemmer()
            # stemmeds = []
            # # Para cada token del texto obtenemos su raíz.
            # for token in tokens:
            # # for token in tokens_nostop:
            #     stemmed = stemmer.stem(token)
            #     stemmeds.append(stemmed)
            # text = nltk.Text(stemmeds)

########################################################################################################################

            # # Lemmatization
            # # Seleccionamos el lematizador.
            # wordnet_lemmatizer = WordNetLemmatizer()
            # lemmatizeds = []
            # nlemmas = []
            # for token in tokens:
            # # for token in tokens_nostop:
            #     lemmatized = wordnet_lemmatizer.lemmatize(token)
            #     lemmatizeds.append(lemmatized)
            # text = nltk.Text(lemmatizeds)
            # # print(text)

########################################################################################################################


            text = nltk.Text(tokens)
            texts.append(text)

    print("Prepared ", len(texts), " documents...")
    print("They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]")

    distanceFunction ="cosine"
    # distanceFunction = "euclidean"
    test = cluster_texts(texts, 4, distanceFunction)
    print("test:      ", test)
    # Gold Standard
    reference = [0, 1, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 3, 3, 0, 1, 2, 0, 1]
    reference_str = '[0 1 2 0 0 0 3 0 0 0 2 0 3 3 0 1 2 0 1]'
    print("reference: ", reference_str)

    # Evaluation
    print("rand_score: ", adjusted_rand_score(reference, test))
