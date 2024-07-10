import numpy as np
import pandas as pd

def data_oulad():
    """
    size = 378
    :return:
    """
    localrep = "./data/oulad/"
    View_inf = pd.read_pickle(localrep + "View_inf.pkl")
    View_seq = pd.read_pickle(localrep + "View_seq.pkl")
    View_seq = View_seq.tolist()
    return View_inf,View_seq

def data_oulad_all():
    """
    size = 5787
    :return:
    """
    localrep = "./data/oulad/"
    View_inf = pd.read_pickle(localrep + "View_inf_all.pkl")
    View_seq = pd.read_pickle(localrep + "View_seq_all.pkl")
    View_seq = View_seq.tolist()
    return View_inf,View_seq

def decode_oulad(x):
    return x

def data_cned():
    localrep = "./data/cned/"
    View_inf = pd.read_pickle(localrep + "View_inf.pkl")
    View_seq = pd.read_pickle(localrep + "View_seq.pkl")
    View_seq = View_seq.tolist()
    return View_inf,View_seq

def decode_cned(x):
    localrep = "./data/cned/"
    names = np.load(localrep + "names.npy",allow_pickle=True)
    t = names.item()
    names_de = {value : key for (key, value) in t.items()}

    seq = eval(x)
    s = []
    for i in seq:
        name = str(names_de[int(int(i)/10000000)])
        num = str(int(int(i)%10000000))
        s.append(name+":"+num)
    return s


def data_covid():
    localrep = "./data/covid/"
    names = [ chr(ord('A')+i) for i in range(26)]
    View_inf = pd.read_pickle(localrep + "View_inf_1000.pkl")
    View_seq = pd.read_pickle(localrep + "View_seq_1000.pkl")
    View_seq = View_seq.tolist()
    return View_inf,View_seq

def data_covid_all():
    localrep = "./data/covid/"
    View_inf = pd.read_pickle(localrep + "View_inf.pkl")
    View_seq = pd.read_pickle(localrep + "View_seq.pkl")
    View_seq = View_seq.tolist()
    return View_inf,View_seq

def decode_covid(x):
    x = eval(x)
    s = []
    for i in x:
        s.append(chr(ord('A')+int(i)))
    return s


import random
import numpy as np
import pandas as pd

def syn(l):
    View_seq = []
    item_list = [1,2,3,4,5]
    i = 0
    while i < l:
        seq = []
        for j in range(3):
            n = random.randint(0, 3)
            itemset = random.sample(item_list, n)
            if len(itemset)!=0:
                seq.append(itemset)
        if len(seq)!=0:
            View_seq.append(seq)
            i+=1
    View_inf = pd.DataFrame([random.randint(1,3) for i in range(l)])
    View_inf.columns = ['A']
    return View_inf,View_seq