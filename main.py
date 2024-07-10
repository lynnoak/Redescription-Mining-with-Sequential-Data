from src.data import *
from src.RMSP import *

import argparse

parser = argparse.ArgumentParser(description='Redescription Mining with sequences')
#parser.add_argument('--sid_n', type=int, help="Number of samples",default=1000)
parser.add_argument('--min_sup', type=float,help="Minimum support ratio of patterns",default=0.1)
parser.add_argument('--min_len', type=int,help="Minimum length of patterns",default=1)
parser.add_argument('--max_gap', type=int,help="Maxmum gaps of patterns",default=3)
parser.add_argument('--max_len', type=int,help="Maxmum length of patterns",default=5)
args = parser.parse_args()

#sid_n = args.sid_n
#seq_mining_para = ["CM-SPAM",0.1,1,5,"",1,True]
seq_mining_para = ["CM-SPAM",args.min_sup,args.min_len,args.max_len,"",args.max_gap,True]

View_inf,View_seq = data_ma31()

print("Data loaded.")

rmsp = RMSP(seq_mining_para = seq_mining_para)
rmsp.fit(View_inf,View_seq,title= "ma31")

import winsound
duration = 1000  # millisecond
freq = 440  # Hz
winsound.Beep(freq, duration)