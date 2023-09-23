import nltk
import sys
import time
import os
import math 
import string 

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

nltk.download('punkt')
nltk.download('stopwords')

""" nltk.download('wordnet')
nltk.download('omw-1.4') """

""" Function to print text with a gradual appearance effect """
def print_with_appearance(text):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()  # Flush the output buffer to make the text appear immediately
        time.sleep(0.03)  # Adjust the sleep duration to control the speed of appearance
    print()  # Move to the next line after printing the complete text

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
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


def tokenize(document):
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
        if token in nltk.corpus.stopwords.words('english'):
            continue

        else:
            all_punct = True
            for char in token:
                if char not in string.punctuation:
                    all_punct = False
                    break

            if not all_punct:
                cleaned_tokens.append(token)

    return cleaned_tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    # Number of documents for idf:
    num_docs = len(documents)

    # Dictionary to count number of docs containing each word:
    docs_with_word = dict()

    # Iterate through documents looking at unique words in each:
    for document in documents:
        doc_words = set(documents[document])

        for word in doc_words:
            if word not in docs_with_word:
                docs_with_word[word] = 1
            else:
                docs_with_word[word] += 1

    # Calculate idfs for each word:
    word_idfs = dict()
    for word in docs_with_word:
        word_idfs[word] = math.log((num_docs / docs_with_word[word]))

    return word_idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    # Dictionary to hold scores for files
    file_scores = {filename:0 for filename in files}

    # Iterate through words in query:
    for word in query:
        # Limit to words in the idf dictionary:
        if word in idfs:
            # Iterate through the corpus, update each texts tf-idf:
            for filename in files:
              tf = files[filename].count(word)
              tf_idf = tf * idfs[word]
              file_scores[filename] += tf_idf

    sorted_files = sorted([filename for filename in files], key = lambda x : file_scores[x], reverse=True)

    # Return best n files
    return sorted_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    # Dict to score sentences:
    sentence_score = {sentence:{'idf_score': 0, 'length':0, 'query_words':0, 'qtd_score':0} for sentence in sentences}

    # Iterate through sentences:
    for sentence in sentences:
        s = sentence_score[sentence]
        s['length'] = len(nltk.word_tokenize(sentence))
        # Iterate through query words:
        for word in query:
            # If query word is in sentence word list, update its score
            if word in sentences[sentence]:
                s['idf_score'] += idfs[word]
                s['query_words'] += sentences[sentence].count(word)

        # Calculate query term density for each sentence:
        s['qtd_score'] = s['query_words'] / s['length']

    # Rank sentences by score and return n sentence
    sorted_sentences = sorted([sentence for sentence in sentences], key= lambda x: (sentence_score[x]['idf_score'], sentence_score[x]['qtd_score']), reverse=True)

    # Return n entries for sorted sentence:
    return sorted_sentences[:n]


if __name__ == "__main__":
    main()
