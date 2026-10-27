#Exercise 4.20
# Choose a different piece of text and ask different questions.
from transformers import pipeline

question_answerer = pipeline('question-answering')
context = """
Transformers are new deep learning neural networks introduced by Hugging Face in 2017, 
mainly used in the field of natural language processing. 
Transformers are based on a concept called attention, a mechanism for weighting different parts of the input based on their importance.
"""

questions = [
    'Who introduced Transformers?',
    'In what year were Transformers introduced?',
    'What concept are Transformers based on?'
]

for question in questions:
    result = question_answerer({
        'question': question,
        'context': context
    })
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}\n")
