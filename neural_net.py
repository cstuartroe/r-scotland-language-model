import math
print('imported math')
#import tensorflow
#print('imported tensorflow')
import numpy as np
print('imported numpy')
from preprocess import *
print('imported preprocess')
import operator
import sys
non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
print('imported sys')
from datetime import datetime
print('imported datetime')

unknowns = []   #it shouldn't need to get used, so I'm resetting unknowns to an empty array
                #so I don't accidentally use it

vocab_size = len(knowns)

word_indices = {}
for i  in range(vocab_size):
    word_indices[knowns[i]] = i

def word2int(word):
    if is_known(word):
        return word_indices[word]
    else:
        return vocab_size
    return vec

def word2vec(word):
    vec = [0] * (vocab_size + 1)
    vec[word2int(word)] = 1
    return vec

def comment2ints(comment):
    assert(comment[0] == '<comment>' and comment[-1] == '</comment>')
    return [word2int(word) for word in comment]

def comment2mtx(comment):
    assert(comment[0] == '<comment>' and comment[-1] == '</comment>')
    return np.array([word2vec(word) for word in comment])

def vec2word(vec):
    if vec[-1] == 1:
        return '<unk/>'
    else:
        return knowns[vec.index(1)]

def softmax(array):
    exps = [math.e**n for n in array]
    total = sum(exps)
    return [n/total for n in exps]

def random_word(prob_dist):
    pd_sum = sum(prob_dist[:-1])
    prob_dist = [item/pd_sum for item in prob_dist[:-1]]
    word = np.random.choice(knowns,p=prob_dist)
    return word

class RNNNumpy:
    def __init__(self, io_size, hidden_size=500, bptt_truncate=4):
        self.io_size = io_size              #size of input and output vectors (vocab size plus one for unknown token)
        self.hidden_size = hidden_size      #size of hidden layer
        self.bptt_truncate = bptt_truncate

        self.U = np.random.uniform(-np.sqrt(1./io_size), np.sqrt(1./io_size), (hidden_size, io_size))
        self.V = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (io_size, hidden_size))
        self.W = np.random.uniform(-np.sqrt(1./hidden_size), np.sqrt(1./hidden_size), (hidden_size, hidden_size))

    #x: an array of ints corresponding to words
    def forward_propagate(self, x):
        steps = len(x)
        #an array of hidden layer vectors for each step, plus one for initial hidden layer value
        s = np.zeros((steps + 1, self.hidden_size))
        #an array of output vectors for each step
        o = np.zeros((steps, self.io_size))

        for step in range(steps):
            from_input = self.U[:,x[step]]
            from_past_hidden = self.W.dot(s[step-1])
            s[step] = np.tanh(from_input + from_past_hidden)
            o[step] = softmax(self.V.dot(s[step]))
        return (o, s)

    def to_sent(self,o):
        out = []
        for vector in o:
            out.append(random_word(vector))
        return out

    #copied below from http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
    #Calculating the loss
    def calculate_total_loss(self, x, y):
        loss = 0
        # For each sentence...
        for i in range(len(y)):
            #print('calculating loss on comment ' + str(i) + ' of ' + str(len(y)))
            (o, s) = self.forward_propagate(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            loss += -1 * np.sum(np.log(correct_word_predictions))
        return loss

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N

    #Backpropagation through time
    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        (o, s) = self.forward_propagate(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    #SGD implementation
    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    # Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs

    # Note: must first determine our equivalent of X_train and y_train
    # I believe we're using rnn = RNNNumpy(vocab_size+1) instead of model = RNNNumpy(vocab_size)
    def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5, batch_size = 100):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            print('--EPOCH: %d--' % epoch)
            # Optionally evaluate the loss
            subset_indices = np.random.choice(len(X_train),batch_size)
            if (epoch % evaluate_loss_after == 0):
                print('performing loss assessment')
                X_subset = [X_train[i] for i in subset_indices]
                y_subset = [y_train[i] for i in subset_indices]
                loss = model.calculate_loss(X_subset, y_subset)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss*1.4427))
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(subset_indices)):
                #print('training round ' + str(i) + ' of ' + str(len(subset_indices)))
                # One SGD step
                model.sgd_step(X_train[subset_indices[i]], y_train[subset_indices[i]], learning_rate)
                num_examples_seen += 1

    #</copied>

    def generate_comment(self):
        s = np.zeros((self.hidden_size))
        comment = ['<comment>']
        i = 0
        
        while comment[-1] != '</comment>':
            if i > 25:
                return ' '.join(comment + ['</comment>'])
            
            from_input = self.U[:,word2int(comment[i])]
            from_past_hidden = self.W.dot(s)
            s = np.tanh(from_input + from_past_hidden)
            new_word = random_word(softmax(self.V.dot(s)))
            comment.append(new_word.translate(non_bmp_map))

            i += 1

        return ' '.join(comment)          

rnn = RNNNumpy(vocab_size+1)
#outputs, hiddens = rnn.forward_propagate(comment2ints(tokenized_unks[0]))
X_train = [comment2ints(comment)[:-1] for comment in tokenized_unks]
y_train = [comment2ints(comment)[1:] for comment in tokenized_unks]

#<copied>
# Note: must first determine our equivalent of X_train and y_train
# Limit to 1000 examples to save time
#print("Expected Loss for random predictions: %f" % np.log(vocab_size))
#print("Actual loss: %f" % rnn.calculate_loss(X_train[:100], y_train[:100]))
#</copied>

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
for i in range(10):
    print(rnn.generate_comment())
print('Training model...')
rnn.train_with_sgd(X_train,y_train)
for i in range(10):
    print(rnn.generate_comment())
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
