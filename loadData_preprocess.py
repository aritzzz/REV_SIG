import os
import glob
import json
#import torch
import numpy as np
from nltk import word_tokenize
from collections import Counter, OrderedDict
#from collections import defaultdict
#from nltk import word_tokenize
#from nltk import sent_tokenize
#from nltk.tokenize import RegexpTokenizer
#from nltk.corpus import stopwords

from Review import Review
from Paper import Paper
from ScienceParse import ScienceParse
from ScienceParseReader import ScienceParseReader

#from sent_encoder import SentenceEncoder

import torchtext
from torchtext.data.example import Example 
from torchtext.data import Dataset, Field, BucketIterator
from torchtext import data

class defaultField(Field):
    def __init__(self, **kwargs):
        super(defaultField, self).__init__(**kwargs)

    """To Do: Also modify the Numericalize function in the Field Class"""

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if name in ['paper', 'review']]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token] + kwargs.pop('specials', [])
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)



class makeExample(Example):

    def __init__(self):
        super(makeExample, self).__init__()


    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            #print(name, field, val)
            if field is not None:
                if isinstance(val, str):
                    val = val.rstrip('\n')
                if name != 'paper' and name != 'review' and name !='id':
                    setattr(ex, name, val)
                else:
                    setattr(ex, name, field.preprocess(val))
        return ex


def preprocess(input, only_char=False, lower=True, stop_remove=False, stemming=False):
  if lower: input = input.lower()
  # if only_char:
  #   tokenizer = RegexpTokenizer(r'\w+')
  #   tokens = tokenizer.tokenize(input)
  #   input = ' '.join(tokens)
  #sents = sent_tokenize(input)
  tokens = word_tokenize(input)
  if stop_remove:
    tokens = [w for w in tokens if not w in stopwords.words('english')]

  # also remove one-length word
  #tokens = [w for w in tokens if len(w) > 1]
  # review = " ".join(tokens)
  # print(sent_tokenize(review))
  return " ".join(tokens)


class sigData(Dataset):

    def __init__(self, examples, fields, **kwargs):
        super(sigData, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.paper), len(ex.review))


    @classmethod
    def getdata(cls, path, fields, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('paper', fields[0]), ('review', fields[1]),\
                     ('recommendation', fields[2]), ('confidence', fields[3])]#, ('id', fields[4])]

        review_dir = os.path.join(path, 'reviews/')
        scienceparse_dir = os.path.join(path, 'parsed_pdfs/')
        paper_json_filenames = glob.glob('{}/*.json'.format(review_dir))

        examples = []
        for paper_json_filename in paper_json_filenames:
            # if paper_json_filename.split('/')[-1] == 'By0ANxbRW.json':
            #id_ = paper_json_filename.split('/')[-1]
            paper = Paper.from_json(paper_json_filename)
            paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID,paper.TITLE,paper.ABSTRACT,scienceparse_dir)
            if paper.SCIENCEPARSE == None:
                continue
            paper_content = preprocess(paper.SCIENCEPARSE.get_paper_content())
            for review in paper.REVIEWS:
                review_content = preprocess(review.COMMENTS)
                rec = review.RECOMMENDATION
                conf = review.CONFIDENCE
                if paper_content != '' and review_content != '':
                    examples.append(makeExample.fromlist(
                        [paper_content, review_content, rec, conf], fields))
        return cls(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path=None, fields = None, train=None,
        validation=None, test=None, **kwargs):

        train_data = None if train is None else cls.getdata(
            os.path.join(path), fields, **kwargs)
        val_data = None if validation is None else cls.getdata(
            os.path.join(path), fields, **kwargs)
        test_data = None if test is None else cls.getdata(
            os.path.join(path), fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                    if d is not None)


if __name__ == "__main__":
    #ID = defaultField(use_vocab=False)#, postprocessing=data.Pipeline(lambda x: x))
    PAPER = defaultField(tokenize = "spacy",
                tokenizer_language="en",
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)
    REVIEW = defaultField(tokenize = "spacy",
                tokenizer_language="en",
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)
    RECOMMENDATION = defaultField(sequential=False,use_vocab=False)
    CONFIDENCE = defaultField(sequential=False,use_vocab=False)

    flds = (PAPER, REVIEW, RECOMMENDATION, CONFIDENCE)#, ID)

    #d = sigData.getdata(path = './Data/2018/', fields = flds)
    d = sigData.splits('./Data/2018/', fields = flds, train='train')
    print(type(d))
    print(type(d[0]))
    print(d[0])
    PAPER.build_vocab(d[0])
    REVIEW.build_vocab(d[0])
    assert len(PAPER.vocab.stoi) == len(REVIEW.vocab.stoi)
    print(len(PAPER.vocab.stoi))
    print(len(REVIEW.vocab.stoi))
    iterators = BucketIterator.splits(
                  d, batch_size=2, shuffle=False)
    for i, b in enumerate(iterators[0]):
        print(b.paper)
        print(b.review)
        print(b.recommendation)
        print(b.confidence)
        break
    # print(PAPER.vocab.stoi)
    # print(REVIEW.vocab.stoi)
    # count=0
    # with open('samples', 'w') as f:
    #     for eg in d:
    #         for key in eg.__dict__.keys():
    #             #print(eg.__dict__[key])
    #             f.write(key + ':' + '\t' + '*'.join(t for t in eg.__dict__[key]) + '\n')
    #         f.write('/////' + '\n')
    #         f.write('\n')
    #         count+=1
    # print('Done with {} reviews'.format(count))