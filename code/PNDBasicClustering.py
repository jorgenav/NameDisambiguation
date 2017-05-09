import re, pprint, os, numpy
import nltk
from nltk.corpus import stopwords
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
        # word_tf.append(collection.tf(word, document))
        # word_tf.append(collection.idf(word))
        word_tf.append(collection.tf_idf(word, document))
    return word_tf

########################################################################################################################

def remove_stop_words(words, ent = False):
    stop_words = set(stopwords.words('english'))
    for word in words:
            if ent:
                if len(re.findall('^#?[a-zA-Z0-9]*', word[0])[0]) > 0:
                    if word[0] in stop_words:
                        words.remove(word)
            else:
                if len(re.findall('^#?[a-zA-Z0-9]*', word[0])[0]) > 0:
                    if word in stop_words:
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
                if len(re.findall('^#?[a-zA-Z0-9]*', word[0])[0]) > 0:
                        stemmized.append((stemmer.stem(word[0]),word[1]))
            else:
                if len(re.findall('^#?[a-zA-Z0-9]*', word[0])[0]) > 0:
                    stemmized.append(stemmer.stem(word))
    return stemmized

########################################################################################################################

def lemmatize(words, ent=False):
    lemmatized = []
    for word in words:
        if ent:
            if len(re.findall('^#?[a-zA-Z0-9]*', word[0])[0]) > 0:
                    lemmatized.append((lemmatizer.lemmatize(word[0]),word[1]))
        else:
            if len(re.findall('^#?[a-zA-Z0-9]*', word[0])[0]) > 0:
                    lemmatized.append(word)
    return lemmatized

########################################################################################################################

# Function to detect entities

java_path = "/bin/java"
os.environ["JAVA_HOME"] = java_path

stanford_dir = "PATH_TO_STANFORD_NER/stanford-ner-2016-10-31/"
jarfile = stanford_dir + "stanford-ner.jar"
modelfile = stanford_dir + "classifiers/english.muc.7class.distsim.crf.ser.gz"

st = StanfordNERTagger(modelfile,jarfile)

stanford_dir = st._stanford_jar[0].rpartition("/")[0]

def entities(sentence):
    tagged_sentence = st.tag(sentence)
    tagged_sentence_new = []
    for tagsen in tagged_sentence:
        if tagsen[0].lower() in ["baker", "roy", "thomas"]:
            lst = list(tagsen)
            lst[1] = 'PERSON'
            tagsen = tuple(lst)
        tagged_sentence_new.append(tagsen)
        if len(re.findall('^#?[a-zA-Z0-9]*', tagsen[0])[0]) == 0:
            tagged_sentence.remove(tagsen)
    return(tagged_sentence_new)

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
    for file in sorted(listing):
        if file.endswith(".txt"):
            url = folder+"/"+file
            f = open(url,encoding="latin-1");
            raw = f.read()
            f.close()
            # Tokenize text and remove punctuation
            # tokens = nltk.word_tokenize(raw)
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(raw)

            # tokens = entities(tokens)
            # tokens = remove_stop_words(tokens, ent=True)
            # tokens = lemmatize(tokens, ent=False)
            tokens = stemmize(tokens, ent=False)
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
