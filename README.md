# arstpy
My homemade python library for multiple purpose
## Usage
Copy the whole arstpy folder to your project folder<br> I'm sorry it is not available on pip yet :c

# matrices 0.1.0
Simple matrix array processing library<br>

## Usage
```python
from arstpy import matrices

print(isMatrix([[1, 2], [3, 4]])) # True
print(isMatrix([[1, 2], [3, 4, 5]])) # False

print(transpose([[1, 2], [3, 4]])) # [[1, 3], [2, 4]]
print(transpose([[1, 2, 3], [4, 5, 6]])) # [[1, 4], [2, 5], [3, 6]]

print(multiply([[1, 2], [3, 4]], [[1, 2], [3, 4]])) # [[7, 10], [15, 22]]

print(inverse([[1, 2], [3, 4]])) # [[-2.0, 1.0], [1.5, -0.5]]

print(divide([[1, 2], [3, 4]], [[1, 2], [3, 4]])) # [[1.0, 1.0], [1.0, 1.0]]

print(add([[1, 2], [3, 4]], [[1, 2], [3, 4]])) # [[2, 4], [6, 8]]

print(subtract([[1, 2], [3, 4]], [[1, 2], [3, 4]])) # [[0, 0], [0, 0]]
```
## Todo Checklist
- [ ] I need to study more about matrices..., or maybe math in general.

 
# chatbotmachinelite beta0.0.1
Stupid Simple Machine Learning ChatBot Engine with Tensorflow-Keras and NLTK Natural Langauge Processing For Lightweight Purpose ChatBot

## Dependencies
- Python >=3.10 (Not tested on lower version)
- Tensorflow & Keras
- NLTK
  - wordnet
  - punkt
  - omw-1.4
- numpy

```bash
pip install tensorflow nltk numpy
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```
## Usage
Refer to `./cbml_example/example.py`
Sample corpus is available in `./cbml_example/corpus.json`
## Todo Checklist
- [ ] Load multiple JSON Corpus(es)
- [ ] Fill the knowledge gaps and minimize ambiguity in the corpus
- [ ] Expand Language Support
- [ ] Increase Code Readability
- [ ] pip Packaging