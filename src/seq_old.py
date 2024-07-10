# Old version for sequential pattern mining for activitiy view
# Based on package PrefixSpan in python


import numpy as np
import pandas as pd
from prefixspan import PrefixSpan

def KMP(text, pattern):
    """
    KnuthMorrisPratt algorithm
    Check if the pattern is a subsequence of text
    :param text: Sequence
    :param pattern: Subsequence
    :return: Ture or False
    """
    pattern = list(pattern)
    length = len(pattern)

    shifts = [1] * (length + 1)
    shift = 1
    for pos, pat in enumerate(pattern):
        while shift <= pos and pat != pattern[pos - shift]:
            shift += shifts[pos - shift]
        shifts[pos + 1] = shift

    startPos = 0
    matchLen = 0
    flag = False
    for c in text:
        while matchLen == length or matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == length:
            flag = True
            break
    return flag

class PS_patterns(object):
    """
    Return the encoded pattern list of text
    :param style: {"all":all patterns,"generator":Generator patterns,"closed":Closed patterns}
    :param Ktop: the number of k
    """
    def __init__(self,style = "generator",Ktop = 5):
        list_style = {"all": [False, False], "generator": [False, True], "closed": [True, False]}
        [self.closed, self.generator] = list_style.get(style, [False, True])
        self.Ktop = Ktop
        self.patterns = []

    def GetPara(self):
        print(self.style)
        print(self.Ktop)

    def GetPattern(self):
        print(self.patterns)
        return self.patterns

    def fit(self,View_seq,Y = None):
        """
        Mining k top patterns by PrefixSpan algorithm
        :param View_seq: the sequences dataset
        :param Y: the labels of the objects
        :return: use GetEncodedView() for the result View_act
        """
        self.patterns = []
        View_seq = list(View_seq)
        if (Y == None):
            ps = PrefixSpan(View_seq)
            ps_K = ps.topk(k=self.Ktop, closed=self.closed, generator=self.generator)
            self.patterns = [i[1] for i in ps_K]

        else:
            group = {}
            for i in range(len(Y)):
                if Y[i] in group.keys():
                    group[Y[i]].append(View_seq[i])
                else:
                    group[Y[i]]=[View_seq[i]]
            shift = 0
            while(len(self.patterns)<self.Ktop):
                pat_cla = []
                for i in group.keys():
                    ps = PrefixSpan(group[i])
                    ps_K = ps.topk(k=self.Ktop*(shift+1), closed=self.closed, generator=self.generator)
                    pat_cla.extend([j[1] for j in ps_K][shift:])
                pat = [x for x in pat_cla if pat_cla.count(x)<len(group.keys())]
                for i in pat:
                    if not i in self.patterns:
                        self.patterns.append(i)
                shift+=1

        self.View_act = np.array([[int(KMP(i,j)) for j in self.patterns] for i in View_seq])
        pattern_names = [ str(i) for i in self.patterns]
        self.View_act = pd.DataFrame(self.View_act,columns=pattern_names)

    def GetEncodedView(self):
        return self.View_act

    def fit_transform(self,View_seq,Y = None):
        self.fit(View_seq,Y)
        return self.View_act
