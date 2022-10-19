# arstpy
My homemade python library for multiple purpose
## Usage
Copy the whole arstpy folder to your project folder<br> I'm sorry it is not available on pip yet :c
### CBML
<details>
<summary></summary>

# chatbotmachinelite beta0.0.1
Stupidly Simple Machine Learning ChatBot Engine with Tensorflow-Keras and NLTK Natural Langauge Processing For Lightweight Purpose ChatBot

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
</details>