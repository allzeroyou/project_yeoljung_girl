import numpy as np
from PIL import Image
import cv2
import pandas as pd
import csv
import os

path = "./final_withouterror/9/"
file_list = os.listdir(path)

#print ("file_list: {}".format(file_list))
#print(file_list[0])
df = pd.DataFrame()
for i in file_list:
    directory = path + i
    #print(directory)

try :
    for i in file_list:
        directory = path + i

        img = Image.open(directory)
        x = np.array(img)
        x = x[0:28, 0:28, 0:1]
        x = x.reshape(784, -1)
        new_df = pd.DataFrame(x)
        new_df = new_df.transpose()
        df = pd.concat([df, new_df])

        # print(df)
except :
    print(i,"에서 오류가 발생했습니다.")

#print(df)
#df2 = pd.read_csv('./mnist_train.csv',encoding='utf-8',index_col=0,engine='python')
df.to_csv('./9.csv',header='false')


"""
    f = open('./7.csv', 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(df)
    f.close()"""


"""
img = Image.open('./fianl/0/1.png')
x = np.array(img)
x = x[0:28, 0:28, 0:1]
x=x.reshape(784,-1)
print(x.shape)
df = pd.DataFrame(x)
df=df.transpose()
f = open('./7.csv', 'a', newline='')
wr = csv.writer(f)
wr.writerow(df)
f.close()

"""

#print(img_path_list[0])
"""matrix = []
y=[]
y.append(0) #인덱스대신 (label명)
for i in range(0,2):
    img = Image.open()
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    x = np.array(img)
   
y=np.array(y)
#print(y.shape)
matrix=[y]
matrix = np.array(matrix)
print(matrix.shape)"""

"""
for i in range(0,172):
    print(img_path_list[i],"\n")
    #img = Image.open(img_path_list[i])
    #x = np.array(img)
    #y = np.array(y)
"""


"""for j in range(28):
    for k in range(28):
        y.append([x[j][k][0]])
=np.array(y)
y=y.reshape(784,-1)
print(y.shape)"""

"""x = x[0:28, 0:28, 0]
print(x)
x = x.reshape(784,1)
print(len(x.shape))
df = pd.DataFrame(x)
df=df.transpose()
print(df)"""

#df.append("7.csv", header=None, index=['0'])
""" f = open('./7.csv', 'a', newline='')
wr = csv.writer(f)
wr.writerow(df)
f.close() 
a=[1,2,3]
b=[4,5,6]
c=[]
c.append(a)
c.append(b)
print(c)}
"""





#df = pd.read_csv('./mnist_test.csv',encoding='utf-8',index_col=0,engine='python')
#print(df)