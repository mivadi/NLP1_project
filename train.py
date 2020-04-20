import torch
from torch.autograd import Variable
import torch.nn as nn
from read_data import *
dtype = torch.FloatTensor


class LSTM_RNN(nn.Module):

    def __init__(self, w2i, i2w, t2i, i2t, l2i, i2l):
        """
        Initialize LSTM RNN.
            :param w2i: dictionary word to index
            :param i2w: dictionary index to word
            :param t2i: dictionary POS-tag to index
            :param i2t: dictionary index to POS-tag
            :param l2i: dictionary label to index
            :param i2l: dictionary index to label
        
        """
        super(LSTM_RNN, self).__init__()
        
        # make dictionaries global
        self.w2i = w2i
        self.i2w = i2w
        self.t2i = t2i
        self.i2t = i2t
        self.l2i = l2i
        self.i2l = i2l
        
        # set information for embeddings
        voca_size = len(self.w2i)
        tag_size = len(self.t2i)
        word_emb_dim = 100
        tag_emb_dim = 25
        
        # create embeddings
        self.word_emb = nn.Embedding(voca_size, word_emb_dim)
        self.tag_emb =nn.Embedding(tag_size, tag_emb_dim)

        # set information for other layers
        self.input_dim = word_emb_dim + tag_emb_dim
        self.hidden_dim = 100 # number of nodes in hidden layers
        self.num_layer = 2 # number of hidden layers
        self.label_dim = len(l2i)
        
        # create LSTM module
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layer)
        
        # hidden layer in LSTM
        self.hidden = self.init_hidden()
        
        # MLP layer for score matrix
        self.hidden2hidden = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.hidden2score = nn.Linear(self.hidden_dim, 1)
        
        # layer for labels
        self.hidden2label = nn.Linear(self.hidden_dim, self.label_dim)
        

    def init_hidden(self):
        """
        Initialize hiddden layers of dimension (num_layers, minibatch_size, hidden_dim).
        """
        return (Variable(torch.zeros(self.num_layer, 1, self.hidden_dim)),
                Variable(torch.zeros(self.num_layer, 1, self.hidden_dim)))

    
    def forward(self, sentence):
        """
        Forward step:
            :param sentence: a list of lists containing index of word and index of POS-tag
        """
        n = len(sentence)
        emb = self.concatenate_emb(sentence)
        hidden_output, self.hidden = self.lstm(emb.view(n, 1, -1), self.hidden)
        
        # create score matrix
        scores = Variable(torch.FloatTensor(n, n).zero_())
        for i in range(n):
            for j in range(n):
                # fill in score matrix
                vi = hidden_output.view(n, -1)[i].view(-1)
                vj = hidden_output.view(n, -1)[j].view(-1)
                viovj= torch.cat([vi,vj]).view(-1)
                scores[i,j] = self.hidden2score(self.hidden2hidden(viovj))
        
        # normalize score matrix
        for j in range(n):
            scores[:,j] = self.softmax(scores[:,j])
        
        # create normalized label matrix
        labels = Variable(torch.FloatTensor(n-1, self.label_dim).zero_())
        for i in range(n-1): # don't search for label of root
            labels[i] = self.softmax(self.hidden2label(hidden_output.view(n, -1)[i+1].view(-1)))
                
        return scores, labels

    
    @staticmethod
    def softmax(column):
        epsilon = 10**(-20)
        return torch.div(torch.exp(column),torch.sum(torch.exp(column))+epsilon)
        
    
    def concatenate_emb(self, sentence):
        """
        Concatenate the word and POS-tag embedding.
            :param sentence: a list of lists containing index of word and index of POS-tag
        """
        for i, word in enumerate(sentence):
            word_emb = self.word_emb(word[0])
            tag_emb = self.tag_emb(word[1])
            if i == 0:
                emb = torch.cat([word_emb.view(-1), tag_emb.view(-1)])
            else:
                emb = torch.cat([emb.view(-1), torch.cat([word_emb.view(-1), tag_emb.view(-1)])])
        return emb

    
def read_data(path, w2i, i2w, t2i, i2t, l2i, i2l):
    """
    Read data and make it ready for usage.
        :param path: path to dictionary
        :param w2i: dictionary word to index
        :param i2w: dictionary index to word
        :param t2i: dictionary POS-tag to index
        :param i2t: dictionary index to POS-tag
        :param l2i: dictionary label to index
        :param i2l: dictionary index to label
    """
    train_input = []
    golden_scores = []
    golden_labels = []
    data_dim = 0
    with open(path, 'r', newline='\n') as file:
        reader = csv.reader(file, delimiter='\t', quotechar=None)
        # create list of sentences
        sentence_in = [[w2i['root'],t2i['ROOT']]]
        # add arc-in root for root -> will be removed in edmonds-algorithm
        sentence_score = [0]
        sentence_label = []
        for row in reader:
            if row[0].split(" ")[0] != "#":
                # each sencente is list of words
                # each word is represented by values:
                # index word, index POS-tag (sentence in)
                # ARC-in, index label (sentence out)
                sentence_in.append([w2i[row[1]],t2i[row[2]]])
                sentence_score.append(int(row[3]))
                sentence_label.append(l2i[row[4]])
            elif data_dim > 0:
                data_dim += 1
                # stop after 1000 sentences
                if data_dim > 1000: break
                # new sentence start
                # put previous sentence is list of sentences
                train_input.append(Variable(torch.LongTensor(sentence_in)))
                golden_scores.append(Variable(torch.LongTensor(sentence_score)))
                golden_labels.append(Variable(torch.LongTensor(sentence_label)))
                sentence_in = [[w2i['root'],t2i['ROOT']]]
                sentence_score = [0]
                sentence_label = []
            else:
                # start of first sentence
                data_dim = 1
                
        train_input.append(Variable(torch.LongTensor(sentence_in)))
        golden_scores.append(Variable(torch.LongTensor(sentence_score)))
        golden_labels.append(Variable(torch.LongTensor(sentence_label)))
        
        return train_input, golden_scores, golden_labels, data_dim-1# dit moet weg....!!! voor de hele data set

    
def train(path, path_out, lr, epochs):
    """
    Create and train a bidirectional LSTM RNN.
        :param path: path to files
        :param lr: learning rate
        :param epochs: number of epochs
    """
    # create dictionaries
    w2i, i2w, t2i, i2t, l2i, i2l = create_dictionaries(path)
                        
    # create model
    model = LSTM_RNN(w2i, i2w, t2i, i2t, l2i, i2l)
    
    # make data ready for usage
    train_input, golden_scores, golden_labels, data_dim = read_data(path, w2i, i2w, t2i, i2t, l2i, i2l)
    
    # define loss function: cross entropy loss
    cross_entropy_loss = nn.CrossEntropyLoss()

    # define optimize method: stochastic gradient descent 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):

        # sentence and target are matrix
        for j in range(data_dim):
            
            # zero gradient buffers
            optimizer.zero_grad()

            # clear hidden
            model.hidden = model.init_hidden()

            # find output of network
            scores, labels = model(train_input[j])
            
            # calculate the loss
            score_loss = cross_entropy_loss(torch.transpose(scores, 0, 1), golden_scores[j])
            label_loss = cross_entropy_loss(labels, golden_labels[j])
            total_loss = score_loss + label_loss
            
            # backpropagate the error
            total_loss.backward()

            # update the weights
            optimizer.step()
            
    torch.save(model, path_out)
    
    return model