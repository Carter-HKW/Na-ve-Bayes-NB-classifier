from cmath import isnan
import pandas as pd
import numpy as np
import copy
read_file = pd.read_csv("D:\AI\project3/train_01.csv")
test_file = pd.read_csv("D:\AI\project3/test_01.csv")
temp_file = copy.copy(test_file)
cat_ipt, count = np.unique(read_file["class"].values, return_counts=True)
result = []
for i in range(len(count)) :
    result.append(count[i]/read_file.shape[0])


for i in read_file.columns :
    read_file[i] = pd.to_numeric(read_file[i], errors='coerce')
    test_file[i] = pd.to_numeric(test_file[i], errors='coerce')
    
    #missing value
    if True in np.isnan(read_file[i]).values:
    
        if float(i) >= 1.0 :
            avg_val = np.nanmean(read_file[i])
            read_file[i] = np.where(np.isnan(read_file[i]), round(avg_val,1), read_file[i])
            test_file[i] = np.where(np.isnan(test_file[i]), round(avg_val,1), test_file[i])

            
        else :
            dis_data = read_file[i].mode().iloc[0]
            read_file[i] = read_file[i].fillna(dis_data)
            test_file[i] = test_file[i].fillna(dis_data)
    if i == "class" :
        break
    #convert continue to discrete
    elif float(i) >= 1 :
        u = np.unique(read_file[i].values)
        read_file[i] = pd.cut(read_file[i], read_file[i].shape[0]//2, labels = range(read_file[i].shape[0]//2))

        temp = pd.Series([u[0], u[-1]])
        temp = pd.concat([test_file[i], temp], ignore_index=True)
        temp =  pd.cut(temp, read_file[i].shape[0]//2, labels = range(read_file[i].shape[0]//2))

        test_file[i] = temp[:-2] 

#classify
for i in range(test_file.shape[0]) :
    #print(i)
    temp_re = copy.copy(result)
    for j in test_file.columns :
        if j == "class" :
            index = result.index(max(result))
            test_file.loc[i, j] = cat_ipt[index]
            #print(np.log(result))
            result = copy.copy(temp_re)
            
        else :
            if float(j) >= 1.0 :
                v = read_file.shape[0]//2
            else :
                u = np.unique(read_file[j].values)
                v = len(u)
            if test_file[j][i] in read_file[j].values :
                ct = pd.crosstab(read_file[j],read_file["class"])
                for k in range(len(count)) :
                    t = ct.loc[test_file[j][i], cat_ipt[k]]
                    
                    result[k] *= (t + 1) / (count[k] + v)
            else :
                for k in range(len(count)) :
                    result[k] *= 1 / (count[k] + v)

test_file["class"] = test_file["class"].astype(int)
#print(test_file["class"].values)
temp_file["class"] = test_file["class"]
temp_file.to_csv("D:\AI\project3/310553037_01.csv")



