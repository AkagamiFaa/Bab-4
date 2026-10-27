#Exercise 4.19
# Choose some different sentences, make sure some are positive and some are negative.
from transformers import pipeline

classifier = pipeline('sentiment-analysis')

sentences = [
    'I really loved the new design of the application!',
    'The food was terrible and I will never go there again.',
    'It was an average experience, nothing special.'
]

results = classifier(sentences)
for sentence, result in zip(sentences, results):
    print(f"Sentence: {sentence}")
    print(f"Result: {result}\n")
