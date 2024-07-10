import random
import numpy as np
import pandas as pd

def get_site_seq(x):
    act_file = x.sort_values(by = 'date')
    Dic_act = []
    k = -1
    last_date = -100
    for i in range(len(act_file)):
        date = act_file.iloc[i]['date']
        iact = act_file.iloc[i]['id_site']
        if (date == last_date):
                Dic_act[k].append(iact)
        else:
            last_date = date
            Dic_act.append([iact])
            k += 1
    return str(Dic_act)

localrep = "./data/oulad/"
presentation = ['2013J','2013B','2014J']
module = ['AAA','BBB']

act = pd.read_csv(localrep+"studentVle.csv")
act = act[act.code_presentation.isin(presentation)]
act = act[act.code_module.isin(module)]
act = act[(act.sum_click>2) & (act.sum_click<50)]
act = act.groupby("id_site").filter(lambda x: (len(x)>500 ) & (len(x)<5000))
act = act.groupby("id_student",as_index=False).apply(get_site_seq)

data = pd.read_csv(localrep+"studentInfo.csv")
data = data[data.code_presentation.isin(presentation)]
data = data[data.code_module.isin(module)]
data["gender"] = data.gender.map({"M":1, "F":0})
data.highest_education = data.highest_education.map({"No Formal quals": 0,
                                                     "Lower Than A Level":1,
                                                     "A Level or Equivalent": 2,
                                                     "HE Qualification": 3,
                                                     "Post Graduate Qualification": 4})
data.age_band = data.age_band.map({"0-35":0, "35-55":35,"55<=":55})
data.disability = data.disability.map({"Y":1, "N":0})
data.final_result = data.final_result.map({"Fail":-1,"Withdrawn":0,"Pass":1,"Distinction":2})
data.imd_band = data.imd_band.map({"0-10%":0,"10-20%":1,"20-30%":2,"30-40%":3,"40-50%":4,"50-60%":5,
                                   "60-70%": 6, "70-80%": 7, "80-90%": 8, "90-100%": 9})
data = data.fillna(0)
data = data.drop(["code_module","code_presentation"],axis = 1)
X = data.merge(act, how='inner', on='id_student')
View_inf = X.iloc[:,1:-1]
View_seq = X.iloc[:,-1].apply(eval)

View_inf.to_pickle(localrep + "View_inf_all.pkl",protocol=4)
View_seq.to_pickle(localrep + "View_seq_all.pkl",protocol=4)
View_inf.to_csv(localrep + "View_inf_all.csv")
View_seq.to_csv(localrep + "View_seq_all.csv")


localrep = "./data/cned/"

View_inf = pd.read_csv(localrep+"user_ma31.csv")

region = {29 :"Bretagne",22 :"Bretagne",35 :"Bretagne",56 :"Bretagne",
          50 :"Normandie",14 :"Normandie",61 :"Normandie",27 :"Normandie",76 :"Normandie",
          62 :"Hauts-de-France",59 :"Hauts-de-France",60 :"Hauts-de-France",80 :"Hauts-de-France",2:"Hauts-de-France",
          8:"Grand Est",51:"Grand Est",10:"Grand Est",55:"Grand Est",52:"Grand Est",
          54:"Grand Est",57:"Grand Est",67:"Grand Est",88:"Grand Est",68:"Grand Est",
          78:"Île-de-France",91:"Île-de-France",77:"Île-de-France",75:"Île-de-France",92:"Île-de-France",93:"Île-de-France",94:"Île-de-France",
          53:"Pays de la Loire",72:"Pays de la Loire",44:"Pays de la Loire",49:"Pays de la Loire",85:"Pays de la Loire",
          5:"Provence-Alpes-Côte d'Azur",4:"Provence-Alpes-Côte d'Azur",83:"Provence-Alpes-Côte d'Azur",13:"Provence-Alpes-Côte d'Azur",
          84:"Provence-Alpes-Côte d'Azur",6:"Provence-Alpes-Côte d'Azur",98:"Provence-Alpes-Côte d'Azur",
          30:"Occitanie",12:"Occitanie",46:"Occitanie",48:"Occitanie",
          82:"Occitanie",32:"Occitanie",65:"Occitanie",81:"Occitanie",
          31:"Occitanie",34:"Occitanie",9:"Occitanie",11:"Occitanie",66:"Occitanie",
          33:"Nouvelle-Aquitaine",40:"Nouvelle-Aquitaine",47:"Nouvelle-Aquitaine",64:"Nouvelle-Aquitaine",
         17:"Nouvelle-Aquitaine",16:"Nouvelle-Aquitaine",24:"Nouvelle-Aquitaine",79:"Nouvelle-Aquitaine",
          86:"Nouvelle-Aquitaine",87:"Nouvelle-Aquitaine",19:"Nouvelle-Aquitaine",23:"Nouvelle-Aquitaine",
          28:"Centre-Val de Loire",41:"Centre-Val de Loire",37:"Centre-Val de Loire",
          45:"Centre-Val de Loire",18:"Centre-Val de Loire",36:"Centre-Val de Loire",
          89:"Bourgogne-Franche-Comté",21:"Bourgogne-Franche-Comté",58:"Bourgogne-Franche-Comté",71:"Bourgogne-Franche-Comté",
          70:"Bourgogne-Franche-Comté",90:"Bourgogne-Franche-Comté",25:"Bourgogne-Franche-Comté",39:"Bourgogne-Franche-Comté",
          1:"Auvergne-Rhône-Alpes",3:"Auvergne-Rhône-Alpes",42:"Auvergne-Rhône-Alpes",69:"Auvergne-Rhône-Alpes",
          15:"Auvergne-Rhône-Alpes",63:"Auvergne-Rhône-Alpes",43:"Auvergne-Rhône-Alpes",7:"Auvergne-Rhône-Alpes",
          26:"Auvergne-Rhône-Alpes",38:"Auvergne-Rhône-Alpes",74:"Auvergne-Rhône-Alpes",73:"Auvergne-Rhône-Alpes",
          20:"Corse"}
View_inf.area_cp = View_inf.area_cp.map(region)
View_inf.to_csv(localrep + "View_inf.csv")
View_inf.to_pickle(localrep + "View_inf.pkl",protocol=4)

View_seq = pd.read_csv(localrep+"sequence_ma31.csv")
X = View_inf.merge(View_seq, how='inner', on='userid')

X.to_pickle(localrep + "X.pkl",protocol=4)
View_inf = X.iloc[:,1:-1]
View_inf.to_csv(localrep + "View_inf.csv")
View_inf.to_pickle(localrep + "View_inf.pkl",protocol=4)

def str2list(x):
    return [[i] for i in x.split(",")[:50]]
View_seq = X.iloc[:,-1].apply(str2list)
View_seq.to_csv(localrep + "View_seq.csv")

t = View_seq.to_list()
names = []
for i in t:
    names.extend(i)
t = names
names = []
for i in t:
    names.extend(i)
names = [''.join(filter(str.isalpha,i)) for i in names]
names = list(set(names))
t = {}
for i in range(len(names)):
    t[names[i]] = i+1
names = t

def str2list_encode(x,names):
    t = x.split(",")[:50]
    s = []
    for i in t:
        name = ''.join(filter(str.isalpha,i))
        num = ''.join(filter(str.isdigit,i))
        mix = names[name]*10000000+int(num)
        s.append([mix])
    return s

View_seq = X.iloc[:,-1].apply(str2list_encode,args = [names])
np.save(localrep + "names.npy",names)
View_seq.to_pickle(localrep + "View_seq.pkl",protocol=4)




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



def mylist(x):
    l = list(x)
    l = [[ord(i)-ord('A')] for i in l]
    return l

def str2list(x):
    x = x.split(",")
    names = ['core', 'mod_scheduler', 'gradereport_overview', 'mod_quiz', 'mod_resource', 'mod_url', 'mod_choice', 'mod_forum', 'mod_glossary', 'mod_dialogue', 'mod_folder', 'mod_page', 'mod_choicegroup', 'mod_wiki', 'mod_scorm', 'mod_lesson', 'mod_book', 'mod_assign', 'mod_chat', 'mod_collaborate']
    l = []
    for i in x:
        p = -1
        try:
            p = names.index(i)
        except:
            continue
        if p!=-1:
            l.append([p])

    return l
