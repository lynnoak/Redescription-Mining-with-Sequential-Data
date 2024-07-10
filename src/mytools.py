from src.data import *
from src.seq import *
from src.RMSP import *
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn import datasets, neighbors, cluster,preprocessing

th_Jac = 0.3
max_p = 0.01
min_sup = 0.2
max_sup = 0.7
max_fail = 10
con_tree = tree.DecisionTreeClassifier(max_depth = 3,class_weight = 'balanced')

View_inf,View_seq = data_cned()
View_inf = View_inf.drop(["area_cp"],axis = 1)
title = "test_cned"
decode_func = decode_cned
seq_mining_para = ["CM-SPAM",0.01,2,5,"",5,True]


View_inf, init_label_atr = atr2initlabel(View_inf)
View_inf_t, View_seq_t = View_inf, View_seq
l8 = round(len(View_inf_t) * 0.8)
View_inf = View_inf_t[:l8]
View_seq = View_seq_t[:l8]
init_label_atr = init_label_atr[:, :l8]



"""
rmsp = RMSP(seq_mining_para = seq_mining_para)
rmsp.fit(View_inf,View_seq)
"""

spd = SPD()

spd.fit(View_seq, seq_mining_para)
init_label_seq = spd.init_label()

init_label = np.r_[init_label_atr, init_label_seq]
time1 = time.time()

redescriptions = pd.DataFrame()
for i in range(len(init_label)):
    count_fail = 0
    count_succ = 0
    maxJac = th_Jac
    description_s = pd.DataFrame()
    label = init_label[i]
    description_a = ""
    count_t = 0
    new_redescription = pd.DataFrame(columns=["description_a", "description_s",
                                              "Jaccard_value", "p_value","All_Jaccard_value",
                                              "sup","SPD","sup_a", "sup_s"])

    # Bulit the tree based decription and match with the sequential patterns
    while (count_fail < max_fail and count_t < 100):
        count_t += 1

        try:
            tree = con_tree.fit(View_inf, label)
        except:
            count_fail += 1
            continue

        description_a = tree2description(tree, View_inf.columns)

        if len(tree.classes_) < 2 or description_a == "":
            t1 = np.random.randint(2, size=len(View_inf))
            t2 = label
            label = np.logical_or(t1, t2).astype(np.int)
            count_fail += 1
            continue

        query_a = np.abs(con_tree.predict(View_inf)).tolist()

        try:
            description_s = spd.pair(query_a)
        except:
            count_fail += 1
            continue

        query_s = description_s['ind']
        Jac = jaccard_score(query_a, query_s)
        p = p_value(query_a, query_s)
        sup = sum(np.logical_and(query_a, query_s))/len(View_inf)

        if (Jac > maxJac and count_succ<5)\
                or (Jac > maxJac and p <= max_p and sup<= max_sup):
                maxJac = Jac
                label = np.array(query_s)
                query_a_t = np.abs(con_tree.predict(View_inf_t)).tolist()
                query_s_t = [check_subseq(description_s['pattern'],mylist2str(i)) for i in View_seq_t]
                Jac_all = jaccard_score(query_a_t, query_s_t)
                new_redescription = pd.DataFrame(
                    [[description_a, str(description_s['pattern']),
                      Jac, p, Jac_all,
                      sup,description_s['pattern'], query_a, query_s]],
                    columns=["description_a", "description_s",
                             "Jaccard_value", "p_value","Gen_jacc",
                             "Support_redescription","SPD","sup_a", "sup_s"]
                             )
                print(Jac)
                print(Jac_all)
                print(new_redescription['description_a'])
                print(new_redescription['description_s'])
                redescriptions = redescriptions.append(new_redescription, ignore_index=True)
                count_fail = 0
                count_succ +=1
        else:
            t1 = np.random.randint(2, size=len(View_inf))
            t2 = np.array(query_s)
            label = np.logical_or(t1, t2).astype(np.int)
            count_fail += 1

    if new_redescription.empty:
        print("Do not found redesciptions.")

if len(redescriptions) == 0:
    print("Find no redescriptions")

time2 = time.time()
redescriptions = redescriptions.sort_values('Jaccard_value', ascending=False).groupby(['description_a','description_s'],as_index=False).first()

redescriptions = redescriptions[redescriptions.p_value<0.01]
FRed = redescriptions.sort_values('p_value', ascending=True).groupby('description_s',as_index=False).first()
FRed = FRed.sort_values('p_value', ascending=True).groupby('description_a', as_index=False).first()


redescriptions.description_s= redescriptions.description_s.apply(decode_func)
redescriptions.to_csv("./result/"+title+" N = " + str(len(View_inf)) + " " + str(seq_mining_para) + " redescriptions.csv")
FRed.description_s= FRed.description_s.apply(decode_func)
FRed.to_csv("./result/"+title+" N = " + str(len(View_inf)) + " " + str(seq_mining_para) + " PMaxRed.csv")
print("Mined " + str(len(spd.patterns))+" SPDs")
print("Mined " + str(len(redescriptions)) + " redescriptions")
print("Mined " + str(len(FRed)) + " redescriptions after filtering")
print("Avg of Jaccard value: " + str(round(redescriptions['Jaccard_value'].mean(), 4)))
print("Avg of p-value: " + str(round(redescriptions['p_value'].mean(), 4)))
print("Avg of genelized Jaccard value: " + str(round(redescriptions['Gen_jacc'].mean(), 4)))
print("Cost time:"+str(time2-time1))
