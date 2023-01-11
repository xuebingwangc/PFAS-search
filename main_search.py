
"""
Created on Mon Oct 12 09:09:35 2020

@author: Bing
"""
import maching_learning as machlea
import  spectrum_featurization as spefea
import pandas as pd
import caculator as ccl
import joblib
import numpy as np
import warnings 
import math
import re

warnings.filterwarnings("ignore")
def formulatolist(formula):
    formula=formula.replace("Cl","A")
    ptn=re.compile('([A-Z])([\d]*)')
    frag=ptn.findall(formula)
    list1=[]
    list2=[]
    list3=[]
    for i in range(len(frag)):
        list1.append(frag[i][0])
        if(frag[i][1]==""):
            list2.append(1)
        else:
            list2.append(int(frag[i][1]))
    if("C" in list1):
        list3.append(list2[list1.index("C")])
    else:
        list3.append(0)
    if("H" in list1):
        list3.append(list2[list1.index("H")])
    else:
        list3.append(0)
    if("A" in list1):
        list3.append(list2[list1.index("A")])
    else:
        list3.append(0)
    if("F" in list1):
        list3.append(list2[list1.index("F")])
    else:
        list3.append(0)
    if("N" in list1):
        list3.append(list2[list1.index("N")])
    else:
        list3.append(0)
    if("O" in list1):
        list3.append(list2[list1.index("O")])
    else:
        list3.append(0)
    if("P" in list1):
        list3.append(list2[list1.index("P")])
    else:
        list3.append(0)
    if("S" in list1):
        list3.append(list2[list1.index("S")])
    else:
        list3.append(0)
    return list3
def listtoformula(list_):
    formula=""
    list2=["C","H","Cl","F","N","O","P","S"]
    for i in range(len(list_)):
        if(list_[i]!=0):
            formula=formula+list2[i]+str(list_[i])
    return formula
def trans_pfas(cac_formula,database_formula,num_thre):
    list_cac=formulatolist(cac_formula)
    list_data=formulatolist(database_formula)
    num=0
    kind=0
    kind_hf=0
    list3=[]
    for i in range(len(list_cac)):
        list3.append(list_cac[i]-list_data[i])
        if(list3[i]!=0):
            if(i!=1 and i!=3):
                num=num+abs(list3[i])
                kind=kind+1
            else:
                num=num+0.5*abs(list3[i])
                kind_hf+=1
            if(num>num_thre):
                return -1,-1,-1,""
    return num,kind,kind_hf,listtoformula(list3)
def transform_pfas1(list_database,df_formula,num_thre=10):
    list_for=[]
    list_error=[]
    list_t=[]
    list_da=[]
    list_n=[]
    list_k=[]
    list_f=[]
    for i in range(len(df_formula)):
        list_data=[]
        list_trans=[]
        list_num=[]
        list_kind=[]
        f=[]       
        for j in range(len(list_database)):
            num_,kind_,f_,trans_=trans_pfas(df_formula["Formula_mole"][i],list_database[j],num_thre)
            if(num_==-1):
                continue
            list_data.append(list_database[j])
            list_trans.append(trans_)
            list_num.append(num_)
            list_kind.append(kind_)
            f.append(f_)
        if(list_data==[]):
            continue
        df_i=pd.DataFrame({"Formula_":list_data,"trans":list_trans,"num":list_num,"kind":list_kind,"f":f,})
        df_i=df_i.sort_values(by=["num","kind","f"],ascending=["True","True","False"])
        df_i=df_i.reset_index(drop=True)
        list_for.append(df_formula["Formula_mole"][i])
        list_error.append(df_formula["Error(ppm)"][i])
        list_t.append(df_i["trans"][0])
        list_da.append(df_i["Formula_"][0])
        list_n.append(df_i["num"][0])
        list_k.append(df_i["kind"][0])
        list_f.append(df_i["f"][0])
    if(list_for==[]):
        return pd.DataFrame()
    df_=pd.DataFrame({"Formula":list_for,"Error(ppm)":list_error,"Database_formula":list_da,"Trans":list_t,"num":list_n,"kind":list_k,"f":list_f})
    df_=df_.sort_values(by=["num","kind","f"],ascending=["True","True","False"])
    df_=df_.reset_index(drop=True)
    return df_

# rank candidate from offline pubchem database
def rank_candidate(train_test,index,df_svm,mz,df_formula,MS2spe,database_pubchem,MS2_error_Da=0.01,inhouse_database=None):
    list_smi=[]
    if(type(database_pubchem)==dict and type(inhouse_database)==dict):
        for i in range(len(df_formula)):
            if(df_formula["Formula_mole"][i] in database_pubchem.keys()):
                list_smi=list_smi+database_pubchem[df_formula["Formula_mole"][i]]
            if(df_formula["Formula_mole"][i] in inhouse_database.keys()):
                list_smi=list_smi+inhouse_database[df_formula["Formula_mole"][i]]
        list_smi=list(set(list_smi))
    elif(type(database_pubchem)==dict and type(inhouse_database)!=dict):
        for i in range(len(df_formula)):
            if(df_formula["Formula_mole"][i] in database_pubchem.keys()):
                list_smi=list_smi+database_pubchem[df_formula["Formula_mole"][i]]
    elif(type(database_pubchem)!=dict and type(inhouse_database)==dict):
        for i in range(len(df_formula)):
            if(df_formula["Formula_mole"][i] in inhouse_database.keys()):
                list_smi=list_smi+inhouse_database[df_formula["Formula_mole"][i]]
    else:
        return pd.DataFrame(),[]                
    
    sim_=np.zeros(shape=(1,len(train_test)))
    MS2=spefea.spetostr(MS2spe)
    for j in range(len(train_test)):
        sim_[0][j]=spefea.FNR(train_test["MSMS spectrum"][j],MS2,train_test["m/z"][j],mz,MS2_error_Da,minmz=50,maxmz=1250)
    fp_pre=np.zeros(shape=(1,len(index)))    
    for k in range(len(index)):
        model_name="model\svm"+str(k)+".model"
        f1=open(model_name,"rb")
        model_svm=joblib.load(f1)
        f1.close()
        fp_pre[0,k]=model_svm.predict(sim_)
    if(list_smi==[]):
        return pd.DataFrame(),fp_pre
    list_score=[]
    for m in range(len(list_smi)):        
        fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi[m])])
        fp_cac=fp_cac[:,index]
        score_=machlea.score(fp_pre,fp_cac,df_svm)
        list_score.append(score_)
    df_fp=pd.DataFrame({"SMILES":list_smi,"score":list_score})
    df_fp=df_fp.sort_values(by="score",ascending=False)    
    df_fp=df_fp.reset_index(drop=True)
    return df_fp,fp_pre
       
# rank along with transformation with offline database
def rank_candidate_2(train_test,index,df_svm,mz,df_formula,MS2spe,database_pubchem,inhouse_database,MS2_error_Da=0.01,thresh=1):
    list_smi=[]
    if(type(database_pubchem)==dict and type(inhouse_database)==dict):
        for i in range(len(df_formula)):
            if(df_formula["Formula_mole"][i] in database_pubchem.keys()):
                list_smi=list_smi+database_pubchem[df_formula["Formula_mole"][i]]
            if(df_formula["Formula_mole"][i] in inhouse_database.keys()):
                list_smi=list_smi+inhouse_database[df_formula["Formula_mole"][i]]
        list_smi=list(set(list_smi))
    else:
        return pd.DataFrame(),""                 
    sim_=np.zeros(shape=(1,len(train_test)))
    MS2=spefea.spetostr(MS2spe)
    for j in range(len(train_test)):
        sim_[0][j]=spefea.FNR(train_test["MSMS spectrum"][j],MS2,train_test["m/z"][j],mz,MS2_error_Da,minmz=50,maxmz=1250)
    fp_pre=np.zeros(shape=(1,len(index)))    
    for k in range(len(index)):
        model_name="model\svm"+str(k)+".model"
        f1=open(model_name,"rb")
        model_svm=joblib.load(f1)
        f1.close()
        fp_pre[0,k]=model_svm.predict(sim_)
    smi_t1=[]
    sco_t1=[]
    trans_t1=[]
    error_ppm=[]
    formula_0=""
    score_for1=0
# caculate transformation score based on inhouse database with threshhold 
    list_database=list(inhouse_database.keys())
    df1=transform_pfas1(list_database,df_formula,num_thre=10)
    if(len(df1)==0):
        df2=pd.DataFrame()
    elif(len(df1)>5):
        df2=df1[0:5].copy()
    else:
        df2=df1.copy()
    for i in range(len(df2)):
        list_smi1=inhouse_database[df2["Database_formula"][i]]
        smi2_=""
        sco1=0 
        for m in range(len(list_smi1)):        
            fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi1[m])])
            fp_cac=fp_cac[:,index]
            score_=machlea.score(fp_pre,fp_cac,df_svm)
            if(score_>sco1):
                smi2_=list_smi1[m]
                sco1=score_
        smi_t1.append(smi2_)
        sco_t1.append(sco1*(1-0.02*(math.pow(df2["kind"][i],2)+df2["num"][i]))*thresh)
        trans_t1.append(df2["Trans"][i])  
        error_ppm.append(str(round(df2["Error(ppm)"][i],2)))
        if(i==0):
            formula_0=df2["Formula"][i]
            score_for1=sco_t1[i]
        else:
            if(sco_t1[i]>score_for1):
                formula_0=df2["Formula"][i]
                score_for1=sco_t1[i]
    
# caculate pubchem smiles score
    sco_t0=[]
    trans_t0=[]
    error_ppm0=[]
    for m in range(len(list_smi)):        
        fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi[m])])
        fp_cac=fp_cac[:,index]
        score_=machlea.score(fp_pre,fp_cac,df_svm)
        sco_t0.append(score_)
        trans_t0.append("")
        error_ppm0.append(str(round((mz-machlea.SmilestoMW(list_smi[m]))*1000000/mz,2)))
    if(sco_t1==[] and sco_t0==[]):
        return pd.DataFrame(),""
    else:
        df_fp=pd.DataFrame({"SMILES":(list_smi+smi_t1),"score":(sco_t0+sco_t1),"trans":(trans_t0+trans_t1),"error":(error_ppm0+error_ppm)})
        df_fp=df_fp.sort_values(by="score",ascending=False)    
        df_fp=df_fp.reset_index(drop=True)
        if(df_fp["trans"][0]==""):
            return df_fp,machlea.SmilestoFormula(df_fp["SMILES"][0])
        else:
            return df_fp,formula_0

# rank along with transformation with offline database excluing true formula
def rank_candidate_valid(train_test,index,df_svm,mz,formula,df_formula,MS2spe,database_pubchem,inhouse_database,MS2_error_Da=0.01,thresh=1):
    list_smi=[]
    if(type(database_pubchem)==dict and type(inhouse_database)==dict):
        for i in range(len(df_formula)):
            if(df_formula["Formula_mole"][i] in database_pubchem.keys() and df_formula["Formula_mole"][i]!=formula):
                list_smi=list_smi+database_pubchem[df_formula["Formula_mole"][i]]
            if(df_formula["Formula_mole"][i] in inhouse_database.keys() and df_formula["Formula_mole"][i]!=formula):
                list_smi=list_smi+inhouse_database[df_formula["Formula_mole"][i]]
        list_smi=list(set(list_smi))
    else:
        return pd.DataFrame()                   
    sim_=np.zeros(shape=(1,len(train_test)))
    MS2=spefea.spetostr(MS2spe)
    for j in range(len(train_test)):
        sim_[0][j]=spefea.FNR(train_test["MSMS spectrum"][j],MS2,train_test["m/z"][j],mz,MS2_error_Da,minmz=50,maxmz=1250)
    fp_pre=np.zeros(shape=(1,len(index)))    
    for k in range(len(index)):
        model_name="model\svm"+str(k)+".model"
        f1=open(model_name,"rb")
        model_svm=joblib.load(f1)
        f1.close()
        fp_pre[0,k]=model_svm.predict(sim_)
        smi_t1=[]
    sco_t1=[]
    trans_t1=[]
    error_ppm=[]
    formula_0=""
    score_for1=0
# caculate transformation score based on inhouse database with threshhold 
    list_database=list(inhouse_database.keys())
    df1=transform_pfas1(list_database,df_formula,num_thre=10)
    if(len(df1)==0):
        df2=pd.DataFrame()
    elif(len(df1)>5):
        df2=df1[0:5].copy()
    else:
        df2=df1.copy()
    for i in range(len(df2)):
        list_smi1=inhouse_database[df2["Database_formula"][i]]
        smi2_=""
        sco1=0 
        for m in range(len(list_smi1)):        
            fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi1[m])])
            fp_cac=fp_cac[:,index]
            score_=machlea.score(fp_pre,fp_cac,df_svm)
            if(score_>sco1):
                smi2_=list_smi1[m]
                sco1=score_
        smi_t1.append(smi2_)
        sco_t1.append(sco1*(1-0.02*(math.pow(df2["kind"][i],2)+df2["num"][i]))*thresh)
        trans_t1.append(df2["Trans"][i])  
        error_ppm.append(str(round(df2["Error(ppm)"][i],2)))
        if(i==0):
            formula_0=df2["Formula"][i]
            score_for1=sco_t1[i]
        else:
            if(sco_t1[i]>score_for1):
                formula_0=df2["Formula"][i]
                score_for1=sco_t1[i]
# caculate pubchem smiles score
    sco_t0=[]
    trans_t0=[]
    error_ppm0=[]
    for m in range(len(list_smi)):        
        fp_cac=np.array([machlea.get_cdk_fingerprints(list_smi[m])])
        fp_cac=fp_cac[:,index]
        score_=machlea.score(fp_pre,fp_cac,df_svm)
        sco_t0.append(score_)
        trans_t0.append("")
        error_ppm0.append(str(round((mz-machlea.SmilestoMW(list_smi[m]))*1000000/mz,2)))
    if(sco_t1==[] and sco_t0==[]):
        return pd.DataFrame(),""
    else:
        df_fp=pd.DataFrame({"SMILES":(list_smi+smi_t1),"score":(sco_t0+sco_t1),"trans":(trans_t0+trans_t1),"error":(error_ppm0+error_ppm)})
        df_fp=df_fp.sort_values(by="score",ascending=False)    
        df_fp=df_fp.reset_index(drop=True)
        if(df_fp["trans"][0]==""):
            return df_fp,machlea.SmilestoFormula(df_fp["SMILES"][0])
        else:
            return df_fp,formula_0
