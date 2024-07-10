#New version for RMSep
#Based on CARTWheel and MuSerCla

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import neighbors
from scipy.stats import binom

from src.seq import *

def p_value(series):
    a = series['sup_inf']
    b = series['sup_act']
    p = sum(a) * sum(b) / (len(a) ** 2)
    return binom.pmf(sum(np.logical_and(a,b)), len(a), p)

def tree2description(tree, names):
    tree_ = tree.tree_
    feature_name = [
        names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    classes = tree.classes_

    def subdescription(node,depth):
        if tree_.feature[node] != -2:
            left_flag, left_together,left_des = subdescription(tree_.children_left[node], depth + 1)
            right_flag, right_together,right_des = subdescription(tree_.children_right[node], depth + 1)


            together = left_together and right_together

            if left_flag!=right_flag:
                together = 0
            elif together==1:
                return left_flag,1,""

            name = feature_name[node]
            threshold = tree_.threshold[node]

            if left_flag == 1:
                t_des = name+" <= " +str(threshold)
                if left_des !="":
                    left_des =  "( "+ t_des+" and " +left_des+" )"
                else:
                    left_des = t_des

            if right_flag == 1:
                t_des = name+" > " +str(threshold)
                if right_des !="":
                    right_des = "( "+t_des+" and " +right_des+" )"
                else:
                    right_des = t_des

            if left_des!="" and right_des!="":
                description = "( "+left_des + " or " +right_des+" )"
            else:
                description = left_des+right_des

            flag = left_flag or right_flag

            return flag,together,description

        else:
            label = np.argmax(tree_.value[node])
            if classes[label] ==1:
                return 1,1,""
            else:
                return 0,1,""

    flag,together,description = subdescription(0,1)
    return description

class remi_seq(object):
    def __init__(self,th_Jac = 0.5,seq_mining_para = ["CM-SPAM",0.5,2,6,"",5,True],tree_para = [3,'balanced'],max_fail = 5):
        self.th_Jac = th_Jac

        self.seq_mining = seqential_pattern(alg = seq_mining_para)
        self.nei_bor = neighbors.NearestNeighbors(metric='jaccard')
        self.patterns = pd.DataFrame()

        self.tree_depth = tree_para[0]
        self.tree_weight = tree_para[1]
        self.con_tree = tree.DecisionTreeClassifier(max_depth = self.tree_depth,class_weight = self.tree_weight)
        self.max_fail = max_fail

        self.redescriptions = pd.DataFrame()

    def get_params(self):
        print(self.max_fail)
        print(self.th_Jac)
        seq_mining_para = self.seq_mining.get_params()
        tree_para = self.con_tree.get_params()
        print(tree_para)
        return {"max_fail":self.max_fail,"th_Jac":self.th_Jac,
                "seq_mining_para":seq_mining_para,"tree_para":tree_para}

    def fit (self,View_inf,View_seq,patterns = pd.DataFrame()):

        if(patterns.empty):
            self.seq_mining.fit(View_seq)
            self.patterns = self.seq_mining.patterns.copy()
        else:
             self.patterns = patterns

        self.nei_bor.fit(np.array(self.patterns['ind'].tolist()))
        init_label_1 = np.array(list(self.patterns.sample(300)["ind"].copy()))
        init_label_2 = np.array(pd.get_dummies(View_inf,columns = View_inf.columns)).T
        init_label = np.r_[init_label_1,init_label_2]
        redescriptions = pd.DataFrame(
            columns=["description_inf", "sup_inf", "description_act", "sup_act", "pattern_ind", "Jaccard_value"])

        for i in range(len(init_label)):
            count_tree = 0
            Jac = 0
            pattern_ind = -1
            label = init_label[i].T
            description_inf = ""
            sup_inf = []
            while (count_tree < self.max_fail):
                Remi_tree = self.con_tree.fit(View_inf, label)
                new_description_inf = tree2description(Remi_tree, View_inf.columns)

                if len(Remi_tree.classes_) < 2 or new_description_inf == "":
                    t1 = np.random.randint(2, size=len(View_inf)).T
                    t2 = label
                    label = np.logical_or(t1, t2).astype(np.int)
                    continue

                predict = self.con_tree.predict(View_inf).tolist()

                t_Jac, t_ind = self.nei_bor.kneighbors([predict])
                t_Jac = 1 - t_Jac[0, 0]
                t_ind = t_ind[0, 0]
                t_patterns = patterns.iloc[t_ind]

                if Jac < t_Jac:
                    label = np.array(t_patterns['ind']).T
                    Jac = t_Jac
                    pattern_ind = t_ind
                    description_inf = new_description_inf
                    sup_inf = predict
                    count_tree = 0
                else:
                    t1 = np.random.randint(2, size=len(View_inf)).T
                    t2 = np.array(t_patterns['ind']).T
                    label = np.logical_or(t1, t2).astype(np.int)
                    count_tree += 1

            if pattern_ind == -1 or description_inf == "":
                continue

            if (pattern_ind in redescriptions.pattern_ind.values) and (
                    description_inf in redescriptions.description_inf.values):
                continue

            if Jac >= self.th_Jac:
                description_act = patterns['pattern'].iloc[pattern_ind]
                sup_act = patterns['ind'].iloc[pattern_ind]
                new_redescription = pd.DataFrame(
                    [[description_inf, sup_inf, description_act, sup_act, pattern_ind, Jac]],
                    columns=["description_inf", "sup_inf", "description_act", "sup_act", "pattern_ind",
                             "Jaccard_value"])
                redescriptions = redescriptions.append(new_redescription, ignore_index=True)
        redescriptions['p_value'] = redescriptions.apply(p_value, axis=1)
        self.redecriptions = redescriptions
        self.JacMaxRed = redescriptions.sort_values('Jaccard_value', ascending=False).groupby('pattern_ind',as_index=False).first()
        self.JacMaxRed = self.JacMaxRed.sort_values('Jaccard_value', ascending=False).groupby('description_inf',as_index=False).first()
        self.PMaxRed = redescriptions.sort_values('p_value', ascending=True).groupby('pattern_ind', as_index=False).first()
        self.PMaxRed = self.PMaxRed.sort_values('p_value', ascending=True).groupby('description_inf',as_index=False).first()

