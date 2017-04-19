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
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import RegexpTokenizer # to remove punctuation


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
        word_tf.append(collection.tf(word, document))
        # word_tf.append(collection.idf(word))
        # word_tf.append(collection.tf_idf(word, document))
    return word_tf

########################################################################################################################

def remove_stop_words(words, ent = False):
    stop_words = set(stopwords.words('english'))
    for word in words:
        if ent:
            # changing manually some entities
            if word[0].lower() in ["baker", "roy", "thomas"]:
                lst = list(word)
                lst[1] = 'PERSON'
                word = tuple(lst)
            if word[0] in stop_words:
                words.remove(word)
        else:
            if word in stop_words:
                words.remove(word)
    return words

########################################################################################################################
            # Function to stemmize the tokens

def stemmize(words, ent=False):
    stemmized = []
    for word in words:
        if ent:
            # changing manually some entities
            if word[0].lower() in ["baker", "roy", "thomas"]:
                lst = list(word)
                lst[1] = 'PERSON'
                word = tuple(lst)

            if word[0].lower() not in ["baker", "roy", "thomas"]:
                stemmized.append((stemmer.stem(word[0]),word[1]))
            else:
                stemmized.append(word)
        else:
            if word.lower() not in ["baker", "roy", "thomas"]:
                stemmized.append(stemmer.stem(word))
            else:
                stemmized.append(word)
    return stemmized

########################################################################################################################

def lemmatize(words, ent=False):
    lemmatized = []
    for word in words:
        if ent:
            # changing manually some entities
            if word[0].lower() in ["baker", "roy", "thomas"]:
                lst = list(word)
                lst[1] = 'PERSON'
                word = tuple(lst)
            if word[0].lower() not in ["baker", "roy", "thomas"]:
                lemmatized.append((lemmatizer.lemmatize(word[0]),word[1]))
            else:
                lemmatized.append(word)
        else:
            if word.lower not in ["baker", "roy", "thomas"]:
                lemmatized.append(lemmatizer.lemmatize(word))
            else:
                lemmatized.append(word)
    return lemmatized

########################################################################################################################

# Function to detect entities

java_path = "/bin/java"
os.environ["JAVA_HOME"] = java_path

stanford_dir = "/home/jkobe/Development/Stanford/stanford-ner-2016-10-31/"
jarfile = stanford_dir + "stanford-ner.jar"
modelfile = stanford_dir + "classifiers/english.muc.7class.distsim.crf.ser.gz"

st = StanfordNERTagger(modelfile,jarfile)

stanford_dir = st._stanford_jar[0].rpartition("/")[0]

def entities(sentence):
    return(st.tag(sentence)) # creo que si le metes una lista de palabras detecta peor las entidades

########################################################################################################################

if __name__ == "__main__":
    folder = "../data"
    # Empty list to hold text documents.
    texts = []
    stop = set(stopwords.words('english'))
    # To stemmize tokens
    stemmer = SnowballStemmer("english")
    # To lemmatize tokens
    lemmatizer = WordNetLemmatizer()

    listing = os.listdir(folder)
    for file in listing:
        if file.endswith(".txt"):
            url = folder+"/"+file
            f = open(url,encoding="latin-1");
            raw = f.read()
            f.close()
            # Tokenize text and remove punctuation
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(raw)
            tokens = entities(tokens)
            tokens = remove_stop_words(tokens, ent=True)
            tokens = lemmatize(tokens, ent=True)
            tokens = stemmize(tokens, ent=True)
            print("TTTTTTTTTTT: ",tokens)
            text = nltk.Text(tokens)

            # print("text: ", text)
            texts.append(text)

    print("Prepared ", len(texts), " documents...")
    print("They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]")

    distanceFunction ="cosine"
    # distanceFunction = "euclidean"
    for t in texts:
        print(t)
    test = cluster_texts(texts, 4, distanceFunction)
    print("test:      ", test)
    # Gold Standard
    reference = [0, 1, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 3, 3, 0, 1, 2, 0, 1]
    reference_str = '[0 1 2 0 0 0 3 0 0 0 2 0 3 3 0 1 2 0 1]'
    print("reference: ", reference_str)

    # Evaluation
    print("rand_score: ", adjusted_rand_score(reference, test))
