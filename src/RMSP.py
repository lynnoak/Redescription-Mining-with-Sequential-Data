#New version for RMSP

import numpy as np
import pandas as pd
from sklearn import tree, neighbors, cluster,preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score
from scipy.stats import binom
import time

from src.seq import *

def p_value(a,b):
    p = sum(a) * sum(b) / (len(a) ** 2)
    return binom.pmf(sum(np.logical_and(a,b)), len(a), p)

def tree2description(tree, names):
    """
    Get the description from the descision tree
    :param tree:
    :param names:
    :return:
    """
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
                t_des = name+" <= " +str(round(threshold, 4))
                if left_des !="":
                    left_des =  "( "+ t_des+" and " +left_des+" )"
                else:
                    left_des = t_des

            if right_flag == 1:
                t_des = name+" > " +str(round(threshold, 4))
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

def atr2initlabel(View_inf):
    cols_num = View_inf._get_numeric_data()
    cols_cat = View_inf.drop(cols_num.columns, axis=1)
    init_label_num = np.random.randint(2, size=len(View_inf)).reshape(-1, 1)
    if len(cols_num.columns) != 0:
        est = preprocessing.KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='kmeans')
        for i in cols_num.columns:
            t = np.array(View_inf[i]).reshape(-1, 1)
            t = est.fit_transform(t)
            init_label_num = np.c_[init_label_num, t]

    if len(cols_cat.columns) != 0:
        cols_cat = pd.get_dummies(cols_cat, columns=cols_cat.columns)
        View_inf = pd.concat([cols_cat, cols_num], axis=1)
        init_label_cols = np.array(cols_cat)
    else:
        init_label_cols = np.random.randint(2, size=len(View_inf)).reshape(-1, 1)

    init_label_atr = np.c_[init_label_num, init_label_cols].T
    return View_inf,init_label_atr

class SPD(object):
    """
    Sequential pattern descriptions and the store of them
    :param reduction: the reducing complexity approach:'KNN':KNN with kd-tree; LSH:LSHForest, PCA:PCA before KNN
    """
    def __init__(self,reduction = 'KNN'):
        self.reduction = reduction
        if reduction == 'KNN' :
            self.nns = neighbors.NearestNeighbors(metric='jaccard')
        if reduction == 'LSH':
            self.nns = neighbors.LSHForest()
        if reduction == 'PCA':
            self.nns = neighbors.NearestNeighbors(metric='cosine')
        if reduction == 'ORI':
            self.nns = None
        self.spd = pd.DataFrame()

    def fit(self,View_seq,seq_mining_para = ["CM-SPAM",0.1,1,5,"",3,True],spd = pd.DataFrame()):
        if(spd.empty):
            seq_mining = seqential_pattern(alg = seq_mining_para)
            seq_mining.fit(View_seq)
            spd = seq_mining.patterns
            if(spd.empty):
                print("No SPD mined")
                exit(1)

        self.spd = spd
        self.patterns = np.array(spd['ind'].tolist())
        print(str(len(self.patterns))+" SPDs mined")

        if self.reduction == 'PCA':
            self.pca = PCA(n_components=10)
            self.pca.fit(self.patterns)
            self.patterns = self.pca.transform(self.patterns)

        if(self.nns != None):
            self.nns.fit(self.patterns)

    def init_label(self):
        if (self.spd.empty):
            print("No SPD mined")
            exit(1)

        clustering = cluster.KMeans(n_clusters=min(int(0.1 * len(self.patterns)) + 1, 50))
        clustering.fit(self.patterns)
        labels = clustering.cluster_centers_
        if self.reduction == 'PCA':
            labels = self.pca.inverse_transform(labels)
        return np.round(labels)

    def pair(self,query):
        if (self.nns == None):
            best = 0
            bestjac = jaccard_score(query,self.patterns[0])
            for i in range(len(self.spd)):
                jac = jaccard_score(query,self.patterns[i])
                if(jac>bestjac):
                    bestjac =best
                    best = i
            return self.spd.iloc[best]
        else:
            query = np.array(query).reshape(1, -1)
            if self.reduction == 'PCA':
                query = self.pca.transform(query)
            dist, ind = self.nns.kneighbors(query, n_neighbors=5)
            return self.spd.iloc[ind[0, 0]]



class RMSP(object):
    """
    Redescirption mining with sequences
    :param th_Jac: Minimum threshold of Jaccard value
    :param seq_mining_para: Name of the seqential pattern mining algorithm and its parameter arguments
    :param tree_para: Max_depth and other arguments of decision tree
    """
    def __init__(self,th_Jac = 0.3,max_p = 0.01,min_sup = 0.05,max_sup = 0.7,seq_mining_para = ["CM-SPAM",0.1,1,5,"",3,True],tree_para = [3,'balanced'],reduction = 'KNN',max_fail = 5):
        self.th_Jac = th_Jac
        self.max_p = max_p
        self.min_sup = min_sup
        self.max_sup = max_sup
        self.max_fail = max_fail

        self.seq_mining_para = seq_mining_para
        self.reduction=reduction
        self.spd = SPD(reduction=reduction)

        self.tree_para = tree_para
        self.tree_depth = tree_para[0]
        self.tree_weight = tree_para[1]
        self.con_tree = tree.DecisionTreeClassifier(max_depth = self.tree_depth,class_weight = self.tree_weight)

        self.redescriptions = pd.DataFrame()


    def get_params(self):
        return {"max_fail":self.max_fail,"min_Jac":self.th_Jac,
                "seq_mining_para":self.seq_mining_para,
                "tree_para":self.tree_para,
                "reduction":self.reduction}

    def fit (self,View_inf,View_seq,spd = pd.DataFrame(), title = ""):

        View_inf, init_label_atr = atr2initlabel(View_inf)
        View_inf_t, View_seq_t = View_inf, View_seq
        l8 = round(len(View_inf_t)*0.8)
        View_inf = View_inf_t[:l8]
        View_seq = View_seq_t[:l8]
        init_label_atr = init_label_atr[:, :l8]

        self.spd.fit(View_seq,self.seq_mining_para,spd)
        init_label_seq = self.spd.init_label()

        init_label = np.r_[init_label_atr, init_label_seq]

        time1 = time.time()
        for i in range(len(init_label)):
            print(i)
            count_fail = 0
            count_succ = 0
            maxJac = self.th_Jac
            description_s = pd.DataFrame()
            label = init_label[i]
            description_a = ""
            count_t = 0
            new_redescription = pd.DataFrame(
                columns=["description_a", "description_s",
                         "Jaccard_value", "p_value","All_Jaccard_value",
                         "sup", "SPD", "sup_a", "sup_s"])

            # Bulit the tree based decription and match with the sequential patterns
            while (count_fail < self.max_fail and count_t < 100):
                count_t += 1

                try:
                    tree = self.con_tree.fit(View_inf, label)
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

                query_a = np.abs(self.con_tree.predict(View_inf)).tolist()

                try:
                    description_s = self.spd.pair(query_a)
                except:
                    count_fail += 1
                    continue

                query_s = description_s['ind']
                Jac = jaccard_score(query_a, query_s)
                p = p_value(query_a, query_s)
                sup = sum(np.logical_and(query_a, query_s)) / len(View_inf)

                if (Jac > maxJac and count_succ < 5) \
                        or (Jac > maxJac and p <= self.max_p and sup <= self.max_sup):
                    maxJac = Jac
                    label = np.array(query_s)
                    query_a_t = np.abs(self.con_tree.predict(View_inf_t)).tolist()
                    query_s_t = [check_subseq(description_s['pattern'], mylist2str(i)) for i in View_seq_t]
                    Jac_all = jaccard_score(query_a_t, query_s_t)
                    new_redescription = pd.DataFrame(
                        [[description_a, str(description_s['pattern']),
                          Jac, p, Jac_all,
                          sup, description_s['pattern'], query_a, query_s]],
                        columns=["description_a", "description_s",
                                 "Jaccard_value", "p_value", "All_Jaccard_value",
                                 "sup", "SPD", "sup_a", "sup_s"]
                    )
                    print(new_redescription['description_a'])
                    print(new_redescription['description_s'])
                    self.redescriptions = self.redescriptions.append(new_redescription, ignore_index=True)
                    count_fail = 0
                    count_succ += 1
                else:
                    t1 = np.random.randint(2, size=len(View_inf))
                    t2 = np.array(query_s)
                    label = np.logical_or(t1, t2).astype(np.int)
                    count_fail += 1

            if new_redescription.empty:
                print("Do not found redesciptions.")

        if len(self.redescriptions) == 0:
            print("Find no redescriptions")
            exit(1)

        time2 = time.time()

        #Filtering the redundancy redescriptions
        self.JacMaxRed = self.redescriptions.sort_values('Jaccard_value', ascending=False).groupby('description_s',as_index=False).first()
        self.JacMaxRed = self.JacMaxRed.sort_values('Jaccard_value', ascending=False).groupby('description_a',as_index=False).first()
        self.PMaxRed = self.redescriptions.sort_values('p_value', ascending=True).groupby('description_s', as_index=False).first()
        self.PMaxRed = self.PMaxRed.sort_values('p_value', ascending=True).groupby('description_a',as_index=False).first()
        self.JacMaxRed.to_csv("./result/"+title+" N = " + str(len(View_inf)) + " " + str(self.seq_mining_para) + " JacMax.csv")
        self.PMaxRed.to_csv("./result/"+title+" N = " + str(len(View_inf)) + " " + str(self.seq_mining_para) + " PMaxRed.csv")
        print("Mined "+str(len(self.JacMaxRed))+" redescriptions")
        n_p = len(self.JacMaxRed['p_value']<0.01)
        print("Mined "+str(n_p)+" redescriptions with p-value below 0.01")
        print("Avg of Jaccard value: " + str(round(self.PMaxRed['Jaccard_value'].mean(), 4)))
        print("Avg of p-value: " + str(round(self.PMaxRed['p_value'].mean(), 4)))
        print("Avg of Jaccard value for all: " + str(round(self.PMaxRed['All_Jaccard_value'].mean(), 4)))
        print("Cost time:" + str(time2 - time1))


