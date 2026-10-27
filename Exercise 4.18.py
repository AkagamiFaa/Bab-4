#Exercise 4.18
# Choose a different piece of text for text analysis
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sentence = """
Python is an interpreted, high-level and general-purpose programming language.
Python's design philosophy emphasizes code readability with its notable use of significant whitespace.
Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.
"""
tokens = nltk.word_tokenize(sentence)
print("Tokens:", tokens)
tagged = nltk.pos_tag(tokens)
print("Tagged:", tagged)
entities = nltk.chunk.ne_chunk(tagged)
print("Entities:", entities)
