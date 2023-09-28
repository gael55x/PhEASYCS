import nltk
import sys
import time
import os
import math
import string
import spacy
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the spaCy NER model
nlp = spacy.load("en_core_web_lg")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

""" 
nltk.download('omw-1.4') """

""" Function to print text with a gradual appearance effect """


def print_with_appearance(text):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()  # Flush the output buffer to make the text appear immediately
        # Adjust the sleep duration to control the speed of appearance
        time.sleep(0.03)
    print()  # Move to the next line after printing the complete text


""" Main program """


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python chatbot.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        # Pass an empty set for named_entities
        filename: tokenize(files[filename], set())
        for filename in files
    }

    # Perform NER on the documents and get named entities
    named_entities = set()
    for filename in files:
        named_entities.update(perform_ner(files[filename]))

    # Prompt user for query
    query = set(tokenize(input("Query: "), named_entities))

    # Calculate IDF values for words based on the query (with attention)
    file_idfs = compute_idfs(file_words, query, named_entities)

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                # Pass named_entities here
                tokens = tokenize(sentence, named_entities)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values for sentences based on the query (with attention)
    idfs = compute_idfs(sentences, query, named_entities)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs,
                            named_entities, n=SENTENCE_MATCHES)
    for match in matches:
        # Concatenate the matches into a single string
        matches_str = ' '.join(matches)

        # Print the concatenated matches with the appearance effect
        print_with_appearance(matches_str)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_dict = dict()

    # Iterate through .txt files in the given directory:
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding="utf8") as file:
                file_string = file.read()
                file_dict[filename] = file_string

    return file_dict


def perform_ner(document):
    """
    Given a document (string), perform Named Entity Recognition (NER) using spaCy.
    Returns a list of named entities found in the document.
    """
    doc = nlp(document)
    entities = [ent.text for ent in doc.ents]
    return entities


def tokenize(document, named_entities):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    cleaned_tokens = []

    # Tokenize document string using nltk
    tokens = nltk.tokenize.word_tokenize(document.lower())

    # Ensure all tokens are lowercase, non-stopwords, non-punctuation
    for token in tokens:
        if token in nltk.corpus.stopwords.words('english') or token in string.punctuation or token in named_entities:
            continue

        else:
            all_punct = True
            for char in token:
                if char not in string.punctuation:
                    all_punct = False
                    break

            if not all_punct:
                # Lemmatize the token
                cleaned_tokens.append(lemmatizer.lemmatize(token))

    return cleaned_tokens


def compute_idfs(documents, query, named_entities):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary. The IDF values will be influenced by the query.
    """

    # Join the tokenized words into strings for each document
    document_strings = [" ".join(words) for words in documents.values()]

    # Create a TfidfVectorizer instance
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the document strings
    tfidf_matrix = tfidf_vectorizer.fit_transform(document_strings)

    # Get the feature names from the vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get the IDF values from the vectorizer
    idf_values = tfidf_vectorizer.idf_

    # Create a mapping of words to IDF values
    word_idfs = {word: idf for word, idf in zip(feature_names, idf_values)}

    # Adjust IDF values based on query attention
    for word in word_idfs:
        word_idfs[word] *= query_attention(word, query, named_entities)

    return word_idfs


def query_attention(word, query, named_entities):
    """
    Calculate the attention score for a word based on its relevance to the query.

    Args:
    word (str): The word to calculate attention for.
    query (set): A set of words representing the query.
    named_entities (set): A set of named entities in the query.

    Returns:
    float: The attention score for the word.
    """
    # Convert the word to lowercase for consistency
    word = word.lower()

    # Calculate word embeddings (you may need to load word vectors or use embeddings from a pre-trained model)
    word_embedding = get_word_embedding(word)

    # Check if the word is part of any named entity in the query
    if any(entity.lower() in word for entity in named_entities):
        return 0.8

    # Calculate cosine similarity between word embedding and query embedding
    query_embedding = get_query_embedding(query)
    similarity_score = calculate_similarity(word_embedding, query_embedding)

    # Use a dynamic threshold for relevance scoring
    if similarity_score > 0.7:
        return similarity_score
    else:
        return 0.001  # Default attention score for non-matching words

def get_word_embedding(word):
    """
    Get the word embedding for a given word.

    Args:
    word (str): The word for which to obtain the embedding.

    Returns:
    numpy.ndarray: The word embedding as a numpy array.
    """
    word = word.lower()  # Ensure lowercase for consistency
    if word in nlp.vocab:
        return nlp.vocab[word].vector
    else:
        # Handle out-of-vocabulary words or return a default embedding
        return nlp.vocab["<unk>"].vector  # You can choose a default word vector for unknown words

def get_query_embedding(query):
    """
    Get the query embedding by averaging word embeddings of query words.

    Args:
    query (set): A set of words representing the query.

    Returns:
    numpy.ndarray: The query embedding as a numpy array.
    """
    # Filter out words not in the vocabulary and obtain their embeddings
    word_embeddings = [get_word_embedding(word) for word in query if word in nlp.vocab]

    if word_embeddings:
        # Calculate the average embedding of query words
        return np.mean(word_embeddings, axis=0)
    else:
        # Handle the case when no valid word embeddings are found in the query
        return np.zeros(nlp.vocab.vectors.shape[1])  # Return a zero vector

def calculate_similarity(embedding1, embedding2):
    # Compute the cosine similarity between two embeddings
    if np.linalg.norm(embedding1) > 0 and np.linalg.norm(embedding2) > 0:
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    else:
        similarity = 0.0  # Handle the case of zero vectors

    return similarity

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    # Dictionary to hold scores for files
    file_scores = {filename: 0 for filename in files}

    # Iterate through words in query:
    for word in query:
        # Limit to words in the idf dictionary:
        if word in idfs:
            # Iterate through the corpus, update each texts tf-idf:
            for filename in files:
                tf = files[filename].count(word)
                tf_idf = tf * idfs[word]
                file_scores[filename] += tf_idf

    sorted_files = sorted([filename for filename in files],
                          key=lambda x: file_scores[x], reverse=True)
    print("Sorted Files:", sorted_files)

    # Return best n files
    return sorted_files[:n]


def top_sentences(query, sentences, idfs, named_entity, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf and query attention.
    """

    # Dict to score sentences:
    sentence_score = {sentence: {'idf_score': 0,
                                 'query_attention_score': 0} for sentence in sentences}

    # Iterate through sentences:
    for sentence in sentences:
        s = sentence_score[sentence]

        # Calculate IDF score for the sentence
        for word in sentences[sentence]:
            if word in idfs:
                s['idf_score'] += idfs[word]

        # Calculate query attention score for the sentence
        for word in query:
            s['query_attention_score'] *= query_attention(
                word, query, named_entity)

    # Rank sentences by combined score (idf + query attention) and return n sentences
    sorted_sentences = sorted([sentence for sentence in sentences], key=lambda x: (
        sentence_score[x]['idf_score'] + sentence_score[x]['query_attention_score']), reverse=True)

    # Return n entries for sorted sentences:
    return sorted_sentences[:n]


if __name__ == "__main__":
    main()

