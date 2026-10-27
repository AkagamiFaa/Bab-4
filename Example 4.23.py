# Example 4.23
# pip install transformers
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
result = classifier('This is a good movie.')
print(result)
