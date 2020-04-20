import csv
from collections import defaultdict, Counter
import os


# dit kan getest worden met de evalution algoritme
def create_CONNL_U(path_in, path_out):
    """
    Create a CONNL-U file with usefull data.
    Input:
        :param path_in: path to file with data from ... 
        :param path_out: path to file for useful data
    """

    # read data from file
    read = []
    with open(path_in, 'r', newline='\n') as file_in:
        reader = csv.reader(file_in, delimiter="\t", quotechar=None)
        # create word counter
        word_count = Counter()
        for row in reader:
            read.append(row)
            if len(row)>1:
                word_count[row[1]] += 1

    # remove file_out if it does exist
    if os.path.isfile(path_out) :
        os.remove(path_out)
    dim = 0
    # create file_out
    with open(path_out, 'w') as file_out:            
        #last_row = ['# nope']
        for row in read:
            
            if len(row) != 0:
                if is_int(row[0]):
                    word = row[1]
                    if word_count[word] == 1:
                        word = '<unk>'
                    file_out.write('{}\t{}\t{}\t{}\t{}\n'.format(row[0], word, row[3], row[6], row[7]))
                elif row[0].split(' ')[0] == '#' and row[0].split(' ')[1] == 'text':
                    file_out.write(row[0]+'\n')
                    dim +=1
                last_row = row
    print(dim)


def is_int(value):
    """
    Check if value represents an integer.
        :param value: string
    """
    try:
        int(value)
        return True
    except ValueError:
        return False
        
    
    
    
def create_dictionaries(path):
    """
    Create dictionaries with words from the CONNL-U file with useful information.
        :param path: path to file with useful information
    """
    
    # create dictionaries
    w2i = defaultdict(lambda: len(w2i))
    i2w = dict()
    t2i = defaultdict(lambda: len(t2i))
    i2t = dict()
    l2i = defaultdict(lambda: len(l2i))
    i2l = dict()
    
    i2w[w2i['root']] = 'root'
    i2t[t2i['ROOT']] = 'ROOT'
    
    with open(path, 'r', newline='\n') as file:
        reader = csv.reader(file, delimiter='\t', quotechar=None)
        for row in reader:
            if row[0].split(" ")[0] != "#":
                # fill in all information to dictionaries
                i2w[w2i[row[1]]] = row[1]
                i2t[t2i[row[2]]] = row[2]
                i2l[l2i[row[4]]] = row[4]
             
    # stop defaultdict behavior 
    w2i = dict(w2i)
    t2i = dict(t2i)
    l2i = dict(l2i)

    return w2i, i2w, t2i, i2t, l2i, i2l