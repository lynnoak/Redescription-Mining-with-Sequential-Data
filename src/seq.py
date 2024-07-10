# Sequential pattern mining for activitiy view
# Based on package SPMF in Java

import numpy as np
import pandas as pd
from spmf import *

def mylist2str(list):
    output = []
    for i in list:
        mystr = ''
        for j in i:
            mystr+= str(j) + ' '
        mystr = mystr[:-1]
        output.append(mystr)
    return output



def check_subseq(lookup_list,input_):
    """
    check if the lookup_list is a subsequence of input_
    :param lookup_list:
    :param input_:
    :return:
    """
    it = iter(input_)
    try:
        for i in lookup_list:
            # read values from the input list until we find one which
            # matches the current value from the lookup list
            while next(it) != i:
                pass
        # we found matches for everything on the lookup list
        return 1
    except StopIteration:
        # we tried to go past the end of the input list
        return 0

class seqential_pattern(object):
    """
    Return the pattern list of text
    :param alg: Name of the seqential pattern mining algorithm and its parameter arguments
    """
    def __init__(self,alg = ["CM-SPAM",0.2,2,5,"",3,True], input_type = "normal"):
        self.algorithm = alg[0]
        self.arguments = alg[1:]
        self.input_type = input_type
        self.title = str(alg)
        self.patterns = []
        self.ViewEncSeq = np.array([])

    def get_params(self):
        print(self.algorithm)
        print(self.arguments)
        return {"algorithm":self.algorithm,"arguments":self.arguments}

    def get_patterns(self):
        print(self.patterns)
        return self.patterns['pattern'].tolist

    def fit(self,View_seq):
        """
        :param View_seq: a list type of the sequences dataset
        :return: use GetPattern for the pd type of patterns
        """
        self.title = "N = "+str(len(View_seq))+" "+self.title

        self.spmf = Spmf(self.algorithm, input_direct=View_seq,input_type=self.input_type,
                    output_filename="./patterns/"+self.title+"_output.txt",
                    arguments=self.arguments,memory=1024)
        self.spmf.run()
        self.patterns = self.spmf.to_pandas_dataframe()

        def enc_seq(x, x_len):
            ind = np.zeros(x_len)
            ind[x] = 1
            return list(ind)

        if 'sid' in list(self.patterns.columns.values):
            self.patterns['ind'] = self.patterns['sid'].apply(enc_seq, args=(len(View_seq),))

    #    self.patterns.to_pickle("./patterns/"+self.title+"_patterns.pkl")
