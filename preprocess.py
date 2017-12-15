from nltk.tokenize import word_tokenize
print('imported tokenize')
import re
print('imported re')
import ngrams #we made this file

known_threshold = 40

def binary_search(key,l):
    low_bound = -1
    high_bound = len(l)
    while True:
        check = (low_bound + high_bound)//2
        if check == low_bound:
            return False
        item = l[check]
        if item == key:
            return True
        elif item < key:
            low_bound = check
        else:
            high_bound = check    

with open('rscotland_corpus.txt','r',encoding = 'utf-8') as f:
    text = f.read()

print('finished reading file')

tokenized = [word_tokenize(re.sub(r'[\*~\^]',r'',comment.lower())) for comment in text.split('\n')]
corpus_word_count = sum(len(sent) for sent in tokenized)
tokenized = [['<comment>'] + comment + ['</comment>'] for comment in tokenized]

print('finished tokenizing')

words = ngrams.ngram(tokenized,1)
print('counted words')

flipped = ngrams.flip_dict(words)
counts = sorted(list(flipped))
print('flipped')

unknowns = sorted([word for i in counts[:known_threshold] for word in flipped[i]])
print('enumerated unknowns')

knowns = sorted([word for i in counts[known_threshold:] for word in flipped[i]])
print('enumerated knowns')

assert(len(unknowns) + len(knowns) == sum(len(value) for key, value in flipped.items()))

def is_known(word):
    return binary_search(word,knowns)

tokenized_unks = [[(word if is_known(word) else '<unk/>') for word in comment] for comment in tokenized]
print('marked unknowns')
