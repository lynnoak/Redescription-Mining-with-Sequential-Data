from src.data import *
from src.seq import *
from src.RMSep import *
from sklearn import tree
from sklearn import neighbors
from scipy.stats import binom

def jac(a,b):
    ab = np.logical_and(a,b)
    ab = sum(ab)
    aob = np.logical_or(a,b)
    aob = sum(aob)
    return ab/aob

X,act_name = data()
View_seq =X['act_seq'].tolist()
drop_inds =['id_student','act_seq']
View_inf =X.drop(drop_inds, axis= 1)

#patterns = pd.read_pickle("CM-SPAM_patterns.pkl")

seq_pat = seqential_pattern(alg = ["CM-SPAM",0.3,1,6,"",3,True])
seq_pat.fit(View_seq)
patterns = seq_pat.patterns.copy()
patterns.to_pickle("CM-SPAM_03_1_6_3.pkl")


th_Jac = 0.5
max_fail = 5
con_tree = tree.DecisionTreeClassifier(max_depth = 3,class_weight = 'balanced')
nei_bor = neighbors.NearestNeighbors(metric= 'jaccard')
nei_bor.fit(np.array(patterns['ind'].tolist()))

redescriptions = pd.DataFrame(columns=["description_inf", "sup_inf", "description_act", "sup_act","pattern_ind","Jaccard_value"])
init_label_1 = np.array(list(patterns.sample(300)["ind"].copy()))
init_label_2 = np.array(pd.get_dummies(View_inf, columns=View_inf.columns)).T
init_label = np.r_[init_label_1, init_label_2]

for i in range(len(init_label)):
    count_tree = 0
    Jac = 0
    pattern_ind = -1
    label = init_label[i].T
    description_inf = ""
    sup_inf = []
    while(count_tree < max_fail):
        Remi_tree = con_tree.fit(View_inf, label)
        new_description_inf = tree2description(Remi_tree, View_inf.columns)

        if len(Remi_tree.classes_)<2 or new_description_inf=="":
            t1 = np.random.randint(2,size = len(View_inf)).T
            t2 = label
            label = np.logical_or(t1,t2).astype(np.int)
            continue

        predict = con_tree.predict(View_inf).tolist()

        t_Jac,t_ind = nei_bor.kneighbors([predict])
        t_Jac = 1-t_Jac[0,0]
        t_ind = t_ind[0,0]
        t_patterns = patterns.iloc[t_ind]

        if Jac<t_Jac:
            label = np.array(t_patterns['ind']).T
            Jac = t_Jac
            pattern_ind = t_ind
            description_inf = new_description_inf
            sup_inf = predict
            count_tree = 0
        else:
            t1 = np.random.randint(2,size = len(View_inf)).T
            t2 = np.array(t_patterns['ind']).T
            label = np.logical_or(t1,t2).astype(np.int)
            count_tree+=1

    if pattern_ind==-1 or description_inf == "":
        print(i)
        print("not found")
        continue

    if (pattern_ind in redescriptions.pattern_ind.values) and (description_inf in redescriptions.description_inf.values):
        print(i)
        print("same")
        continue

    if Jac >= th_Jac:
        print(i)
        print(description_inf)
        description_act = patterns['pattern'].iloc[pattern_ind]
        print(description_act)
        sup_act = patterns['ind'].iloc[pattern_ind]
        new_redescription = pd.DataFrame([[description_inf,sup_inf,description_act, sup_act,pattern_ind,Jac]],
                                         columns = ["description_inf","sup_inf","description_act", "sup_act","pattern_ind","Jaccard_value"])
        redescriptions = redescriptions.append(new_redescription, ignore_index=True)
redescriptions['p_value'] = redescriptions.apply(p_value,axis = 1)

JacMaxRed = redescriptions.sort_values('Jaccard_value', ascending=False).groupby('pattern_ind', as_index=False).first()
PMaxRed = redescriptions.sort_values('p_value', ascending=True).groupby('pattern_ind', as_index=False).first()

