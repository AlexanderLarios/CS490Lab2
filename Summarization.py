import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('wordnet')
nltk.download('punkt')

# Summary: import txt file, remove punctuation
# Returns: check: string containing contents of txt file with out common punctuation
def ImportFile():
    file_in = open("wiki.txt", 'r')
    # Remove as much punctuation as possible (makes better bigrams)
    punct = ".,;:''?``"
    replace = "         "
    trans = str.maketrans(punct, replace)
    text = file_in.read()
    check = text.translate(trans)

    # Return text to be tokenized
    return check

# Summary: tokenize the text
# Param: file_text - string to be tokenized
# Returns: word_tokens
def CreateTokens(file_text):
    word_Tokens = nltk.word_tokenize(file_text)
    return word_Tokens

# Summary: lemmentize the word tokens, output the results to a file
# Param: words - tokens
def Lemmentize_Words(words):
    lemmentizer = WordNetLemmatizer()
    output_file = open("lemmatized.txt", 'w')
    for token in words:
        output_file.write(lemmentizer.lemmatize(token) + '\n')

# Summary: creates bigrams- counts them - finds 5 most common bigrams - find sentences containing most common
# bigrams and output them to a file.
# Param: words - tokenized text
def Bigrams(words):
    # File Management
    output_file = open("bigram.txt", 'w')
    output_file2 = open("bigram_count.txt", 'w')
    output_file3 = open("sentences", 'w')
    input_file = open("wiki.txt", 'r')
    file_in = input_file.read()

    # Create bigrams
    bigram_tokens = nltk.bigrams(words)
    # convert to tuple for easier use
    list_bigram = tuple(bigram_tokens)

    # Output full list of bigrams to file
    for bigram in list_bigram:
        output_file.write(str(bigram) + '\n')

    # Count the frequency of each bigrm
    bigram_counter = Counter(list_bigram)

    # Output the counts to a file
    for item ,count in bigram_counter.items():
        output_file2.write(str(item) + ' ' + str(count) + '\n')
    output_file2.write("\n\n\n")
    output_file2.write("Most frequent bigrams:" + '\n')

    # Output the top 5 most frequent bigrams & output
    freq_bigrams = tuple(bigram_counter.most_common(5))
    for list_item in freq_bigrams:
        output_file2.write(str(list_item) + '\n')

    # Return only the bigram (strip the count)
    search_bigrams = []
    for f, f2 in freq_bigrams:
        search_bigrams.append(tuple(f))

    # Search the original text for the bigrams, make list of the sentences
    sentences = []
    check = 0
    for sent in file_in.split('.'):
        check += 1
        for item_a, item_b in search_bigrams:
            test = str(item_a) + ' ' + str(item_b)
            if test in sent:
                sentences.append(sent)

    # Print the sentences to a file
    for s in sentences:
        output_file3.write(str(s))


text_file = ImportFile()
tokens = CreateTokens(text_file)
Lemmentize_Words(tokens)
Bigrams(tokens)


