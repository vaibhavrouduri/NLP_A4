import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    stopwords = [
        'a','about','above','after','again','against','all','am','an','and','any',
        'are','as','at','be','because','been','before','being','below','between',
    'both','but','by','during','each','few','for','from','further','had','has',
        'have','having','he','her','here','hers','herself','him','himself','his',
        'how','i','if','in','into','is','it','its','itself','me','more','most',
        'my','myself','now','of','off','on','once','only','or','other','our',
        'ours','ourselves','out','over','own','same','she','so','some','such',
        'than','that','the','their','theirs','them','themselves','then','there',
        'these','they','this','those','through','to','too','under','until','up',
        'very','was','we','were','what','when','where','which','while','who',
        'whom','why','will','with','you','your','yours','yourself','yourselves'
    ]

    example["text"] = example["text"].lower()
    example_tokens = word_tokenize(example["text"])

    example_tokens = [ t for t in example_tokens if not (t in stopwords and random.random() < 0.5)]

    new_tokens = []
    for token in example_tokens:
      if token.isdigit():
        new_tokens.append(token)
        continue
      synonyms = set()
      synsets = wordnet.synsets(token)
      if synsets:
        for lemma in synsets[0].lemmas():
          synonyms.add(lemma.name().replace("_", " "))

      synonyms.discard(token)

      synonyms = [s for s in synonyms if s.isalpha() and len(s) > 2]

      if synonyms and random.random() < 0.20:
        new_tokens.append(random.choice(list(synonyms)))
      else:
        new_tokens.append(token)

    example["text"] = " ".join(new_tokens)






    # raise NotImplementedError

    ##### YOUR CODE ENDS HERE ######

    return example
