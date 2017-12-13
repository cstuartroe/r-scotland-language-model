import math
print('imported math')
import tensorflow
print('imported tensorflow')
import numpy as np
print('imported numpy')
from preprocess import *

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
    return [word2int(word) for word in comment[1:-1]]

def comment2mtx(comment):
    assert(comment[0] == '<comment>' and comment[-1] == '</comment>')
    return [word2vec(word) for word in comment[1:-1]]

def vec2word(vec):
    if vec[-1] == 1:
        return '<unk/>'
    else:
        return knowns[vec.index(1)]

def softmax(array):
    exps = [math.e**n for n in array]
    total = sum(exps)
    return [n/total for n in exps]

def random_word(prob_dist,exclude_unks = False):
    word = np.random.choice(knowns + ['<unk/>'],p=prob_dist)
    return word

class RNNNumpy:
    def __init__(self, io_size, hidden_size=100, bptt_truncate=4):
        self.io_size = io_size              #size of input and output vectors (vocab size plus one for unknown token)
        self.hidden_size = hidden_size      #size of hidden layer
        self.bptt_truncate = bptt_truncate  #idk yet

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
            s[step] = np.tanh(self.U[:,x[step]] + self.W.dot(s[step-1]))
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
        for i in np.arange(len(y)):
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

    RNNNumpy.calculate_total_loss = calculate_total_loss
    RNNNumpy.calculate_loss = calculate_loss

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

    RNNNumpy.bptt = bptt

    #Gradient checking
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error &gt; error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)

    RNNNumpy.gradient_check = gradient_check
    '''
    #temporarily commented out:
    # To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
    grad_check_vocab_size = 100
    np.random.seed(10)
    model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
    model.gradient_check([0,1,2,3], [1,2,3,4])
    '''

    #SGD implementation
    # Performs one step of SGD.
    def numpy_sdg_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    RNNNumpy.sgd_step = numpy_sdg_step

    # Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs

    # Note: must first determine our equivalent of X_train and y_train
    # I believe we're using rnn = RNNNumpy(vocab_size+1) instead of model = RNNNumpy(vocab_size)
    def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = model.calculate_loss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                # Adjust the learning rate if loss increases
                if (len(losses) &gt; 1 and losses[-1][1] &gt; losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                model.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1

    #</copied>

rnn = RNNNumpy(vocab_size+1)
outputs, hiddens = rnn.forward_propagate(comment2ints(tokenized_unks[0]))

#<copied>
# Note: must first determine our equivalent of X_train and y_train
# Limit to 1000 examples to save time
print "Expected Loss for random predictions: %f" % np.log(vocab_size)
print "Actual loss: %f" % rnn.calculate_loss(X_train[:1000], y_train[:1000])
#</copied>
