#Old version for RMSep
#Based on CARTWheel and MuSerCla

import numpy as np
import pandas as pd
from sklearn import tree

from src.seq_old import *

class remi_tree_seq(object):
    def __init__(self,remi_style = 0,max_fail = 5,th_Jac = 0.5,seq_mining_para = ["generator",5],tree_para = [3]):
        self.remi_style = remi_style
        self.max_fail = max_fail
        self.th_Jac = th_Jac

        self.seq_mining_style = seq_mining_para[0]
        self.seq_mining_ktop = seq_mining_para[1]
        self.seq_mining = PS_patterns(style=self.seq_mining_style, Ktop=self.seq_mining_ktop)

        self.tree_depth = tree_para[0]
        self.con_tree = tree.DecisionTreeClassifier(max_depth = self.tree_depth)


        self.patterns = []
        self.redecptions = []
        self.View_inf = pd.DataFrame([])
        self.View_inf_name = []
        self.View_seq = []
        self.View_act = pd.DataFrame([])
        self.View_act_name =[]
        self.flag_InfLeft = True
        self.log = []

    def Clean(self):
        self.patterns = []
        self.redecptions = []
        self.View_inf = pd.DataFrame([])
        self.View_inf_name = []
        self.View_seq = []
        self.View_act = pd.DataFrame([])
        self.View_act_name =[]
        self.flag_InfLeft = True
        self.log = []

    def get_params(self):

        return 0

    def paths_class(self,X_test,Remi_tree,name):
        X_test =np.array(X_test)
        node_ind = Remi_tree.decision_path(X_test)
        leaf_id = Remi_tree.apply(X_test)
        n_node_samples = Remi_tree.tree_.n_node_samples
        feature = Remi_tree.tree_.feature
        threshold = Remi_tree.tree_.threshold
        classse = []
        classes_sup = []
        for sample_id in range(len(X_test)):
            node_index = node_ind.indices[node_ind.indptr[sample_id]:
                                                node_ind.indptr[sample_id + 1]]
            sample_path = ""
            for node_id in node_index:
                # continue to the next node if it is a leaf node
                if leaf_id[sample_id] == node_id:
                    classes_sup.append(n_node_samples[node_id])
                    continue
                # check if value of the split feature for sample 0 is below threshold
                if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"

                sample_path+=name[feature[node_id]]+threshold_sign+"{0:.2f}".format(threshold[node_id])+" & "

            classse.append(sample_path)

        return classse,classes_sup

    def eval_Jac(self,description_a,support_a,description_b,support_b):
        n = len(description_a)
        redescption = []
        support = {}
        for sample_id in range(n):
            new_redescption = description_a[sample_id] + " ~ "+ description_b[sample_id]
            if (new_redescption in redescption):
                support[new_redescption][0] +=1
            else:
                redescption.append(new_redescption)
                support[new_redescption] = [1,support_a[sample_id],support_b[sample_id]]

        redescption_Jac = []
        for new_redescption in redescption:
            if (support[new_redescption][0]/
                    (support[new_redescption][0]+support[new_redescption][0]-support[new_redescption][0])
                    >=self.th_Jac):
                redescption_Jac.append(new_redescption)

        return redescption_Jac

    def check_red(self):
        return 0

    def fit (self, X, act_ind = 'act_seq', drop_inds =['id_student']):

        if(not isinstance(X, pd.DataFrame)):
            print("warning: X need to be a pandas DataFrame")
            try:
                X= pd.DataFrame(X)
            except:
                return 0

        drop_inds.append(act_ind)

        self.Clean()
        log_count = 0
        log_temp = {}

        self.View_inf =X.drop(drop_inds, axis= 1)
        self.View_inf_name = list(self.View_inf.columns)
        self.View_seq = list(X[act_ind])
        print(self.View_inf.iloc[0])
        print(self.View_seq[0])

        self.seq_mining.fit(self.View_seq)
        self.patterns = self.seq_mining.GetPattern()
        self.View_act = self.seq_mining.GetEncodedView()
        self.View_act_name = list(self.View_act.columns)
        print(self.View_act.iloc[0])

        log_temp['count'] = log_count
        log_temp['pattern'] = self.patterns
        log_temp['redescription'] = self.redecptions
        self.log.append(log_temp)


        feature_side = self.View_inf
        classes_init = (2*np.random.random(len(feature_side))).astype(int)

        Remi_tree = self.con_tree.fit(feature_side,classes_init)
        classes,classes_sup = self.paths_class(feature_side,Remi_tree,name = feature_side.columns)

        count = 0
        self.flag_InfLeft = False


        while(count<self.max_fail):
            if(self.flag_InfLeft):
                feature_side = self.View_inf
                log_count += 1
                log_temp['count'] = log_count
                log_temp['pattern'] = self.patterns
                log_temp['redescription'] = self.redecptions
                self.log.append(log_temp)
            else:
                if(self.remi_style==0):
                    feature_side = self.View_act
                else:
                    self.seq_mining.fit(self.View_seq,classes)
                    self.patterns = self.seq_mining.GetPattern()
                    self.View_act = self.seq_mining.GetEncodedView()
                    self.View_act_name = list(self.View_act.columns)
                    feature_side = self.View_act

            Remi_tree = self.con_tree.fit(feature_side,classes)
            new_classes,new_classes_sup = self.paths_class(feature_side,Remi_tree,name=feature_side.columns)
            new_redescption = self.eval_Jac(classes,classes_sup,new_classes,new_classes_sup)
            classes = new_classes
            classes_sup = new_classes_sup

            if (len(set(self.redecptions)-set(new_redescption))==0):
                count += 1
            else:
                count = 0

            self.redecptions = list(set(self.redecptions.extend(new_redescption)))
            self.flag_InfLeft = not self.flag_InfLeft

        self.check_red()

    def GetRemi(self):
        return self.redecptions

    def fit_transform(self,X,act_ind = 'act_seq',drop_inds =['id_student']):
        self.fit(X, act_ind = act_ind, drop_inds=drop_inds)
        return self.redecptions


















