import torch
from torch.autograd import Variable
from edmonds_algorithm import *
from read_data import *


def dependency_parser(sentence_index, sentence, network, dicts):
    """
    Find dependency parser of a sentence.
        :param sentence_index: [[index_word_0, index_postag_0], ... , [index_word_n, index_postag_n]]
        :param sentence: is the actual sentence (string)
        :param network: path to pytorch network
        :param dicts: path to CONLL-U file
    """
    w2i, i2w, t2i, i2t, l2i, i2l = create_dictionaries(dicts)
    
    n = len(sentence)
    
    #make torch variable of the sentence
    Sent = Variable(torch.LongTensor(sentence_index))
    
    #load in network
    parser = torch.load(network)
    
    #calculate score-matric and label-matric
    score, label = parser(Sent)
    score_matrix = score.data.numpy()
    
    #edmonds
    heads = edmonds(score_matrix)
    
    label_matrix = label.data.numpy()

    list_of_labels = []
    for i in range(len(label_matrix)):
        highest = np.argmax(label_matrix[i])
        label = i2l[highest]
        list_of_labels.append(label)
        
    # the tree that is made is minus the root. 
    tree = [] 
    for i in range(1,n):
        tree.append([sentence[i], heads[i], list_of_labels[i-1]])
        
    return tree


def compare(output, golden_tree, n):
    """
    Calculate the AUS and LAS score of each sentence.
        :param output: predicted output
        :param golden_tree:  the known tree.
        :param n: the lenght of the sentence
    """
    UAS = 0 
    LAS = 0 
    wrong = 0
    
    for i in range(len(output)):
        out = output[i]
        tree_word = golden_tree[i]
       
        # if a word has the right arc:
        if int(out[1]) == int(tree_word[1]):
            UAS += 1
            
            #If this arc also has the right label
            if out[2] == tree_word[2]:
                LAS +=1
        else:
            wrong += 1
        
    return UAS, LAS


def test_accuracy(datafile, network, dicts):
    """
    Test the accuracy of the dependacy parser.
        :param datafile: path test data
        :param network: path to pytorch network
        :param dicts: path to CONLL-U file
    """
    w2i, i2w, t2i, i2t, l2i, i2l = create_dictionaries(dicts)

    #open de dataset.
    with open(datafile, 'r', newline='\n') as file_in:
        reader = csv.reader(file_in, delimiter="\t", quotechar=None)
        
        #add root to the sentences. 
        sentence_in = [[w2i["root"],t2i["ROOT" ]]]
        sentence = ["root" ] 
        
        #define variables
        tree = []   
        AUS = 0
        LAS = 0
        word_count = 0
        amount_of_sentences = 0
        
        for row in reader:
            if amount_of_sentences != 500:
                if len(row ) > 1 and row[0] != '8.1': 
                    word = row[1]
                    sentence.append(word)
                    if word in w2i:
                        index_word = w2i[word]
                    else:
                        #word is unknown
                        word = "<unk>"
                        index_word = w2i[word]
                    
                    postag = row[3]
                    index_postag = t2i[postag]
                    # from this sentence the predicited tree is predicted
                    sentence_in.append([index_word, index_postag])

                    # This will be the golden tree
                    tree.append([word, row[6], row[7]])

                if len(row) == 0: 
                    # then the end of a line is found.
                    predicted_tree = dependency_parser(sentence_in, sentence, network, dicts)
                    word_count += len(predicted_tree)
                    amount_good_arcs, good_arcs_labels = compare(predicted_tree, tree, len(tree))
                    AUS += amount_good_arcs
                    LAS += good_arcs_labels
            
                    # set varibles empty and ready for analyzing new sentence.
                    sentence_in = [[w2i["root" ],t2i["ROOT" ]]]
                    tree = []
                    sentence = ["root"]
                    amount_of_sentences += 1
            else:
                per_AUS = AUS/word_count *100
                per_LAS = LAS/word_count *100

                return per_AUS, per_LAS