import numpy as np
import pandas as pd
import csv
import os
import sys
from scipy import stats




alphas = [0.0001,0.001]
hiddenSize = 4

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)



def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

'''X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])'''
faultylno = int(raw_input("Enter faulty line number "));
last=-1
fp=open("print_tokens2v4.c.gcov");
for line in fp:
	if "lcount" in line:
		last=line.split(",")[0].split(":")[1];
for alpha in alphas:
       print "\nTraining With Alpha:" + str(alpha)
	#dataset
       df_train=pd.read_csv('funcBranchv4.csv')

	#training output dataset
       y = np.array([df_train['Result']]).T

	#training input dataset
       df_train.drop(['Result'],1 , inplace=True)
       t_in = df_train.values.tolist()
       X = np.array(t_in)

       rank_sus=[]
       #vr=np.identity(len(X.T))
       #VS =vr
       df_test=pd.read_csv('virtualBranch.csv')
       test_in = df_test.values.tolist()
       VS = np.array(test_in)
       np.random.seed(1)
       print("length of x"+str(len(X.T)))
       print("length of x..."+str(len(y.T)))
    # randomly initialize our weights with mean 0
       synapse_0 = 2*np.random.random((len(X.T),hiddenSize)) - 1
       synapse_1 = 2*np.random.random((hiddenSize,1)) - 1
 
       for j in xrange(500):
            #print("calculating")
        # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = sigmoid(np.dot(layer_0,synapse_0))
            layer_2 = sigmoid(np.dot(layer_1,synapse_1))
            #layer_3 = sigmoid(np.dot(layer_2,synapse_2))
            #layer_3_error = layer_3 - y
            #layer_3_delta = layer_3_error*sigmoid_output_to_derivative(layer_3)
            layer_2_error = layer_2 - y
            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
            layer_1_error = layer_2_delta.dot(synapse_1.T)
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
            #synapse_2 -= alpha * (layer_2.T.dot(layer_3_delta))
            synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
            synapse_0 -= alpha * np.dot(zip(*layer_0),(layer_1_delta))
       a = sigmoid(np.dot(VS,synapse_0))
       b = sigmoid(np.dot(a,synapse_1))
       #c = sigmoid(np.dot(b,synapse_2))
       print("length of b"+str(len(b)))
       #print(b)
       array=b.flatten()
       array=-array
       temp = array.argsort()
       ranks = np.empty_like(temp)
       ranks[temp] = np.arange(len(array))
       #print(ranks)
       rank_sus=np.append(rank_sus,ranks)
       #print(rank_sus)
       datalist = []
       with open('funcBranchv4.csv', 'rt') as csvfile:
        	readobj = csv.reader(csvfile, delimiter = ',')
		#heading=next(readobj)
	        for row in readobj:	
	        	rowlist = []
	        	for element in row:		
				rowlist.append((element))
			datalist.append(rowlist)
       funcdata=[]
       elementinrow=len(datalist[0])
       elementincol=len(datalist)
       #print(elementinrow, elementincol)
       count=1
       k=0
       #for y in range(elementinrow-1):
       tag=0
       while (k<elementinrow-1):
		x=k
		occur=[]
		count=1
		startline=int(datalist[0][x].split(":")[1])
		while datalist[0][x]==datalist[0][x+1]:
			count=count+1
			x=x+1
		k=x+1
		if (k<elementinrow-1):		
			endline=int(datalist[0][k].split(":")[1])-1
		startindex=tag
		endindex=tag+count-1
		tag=tag+count
		occur.append(startindex)
		occur.append(endindex)
		occur.append(startline)
		occur.append(endline)
		occur.append(count)
		funcdata.append(occur)
       #print(funcdata)
       l1=len(funcdata)
       funcdata[l1-1][3]=int(last)
       #print(l1)
       for x in range(l1):
		n=funcdata[x][1]-funcdata[x][0]+1
		count=0
		for i in range(funcdata[x][0],funcdata[x][1]+1):
			count=count+b[i][0]
		avg=count/float(n)
		funcdata[x].append(avg)
       #print(funcdata)
       sorted_func = sorted(funcdata,key=lambda l:l[3], reverse = True)
       #print(" \n")
       #print("after sorting:")
       for i in range(len(sorted_func)):
	       sorted_func[i].append(i+1)
       #print (sorted_func)
       #print(" \n")
       exereal=[]
       with open("with-line-num.txt") as textfile1: 
	       for y in textfile1:
			flag=y.split(":")[0]
	                if "-" in flag:
				continue
			else:
				stmtinfo=[]
				f=y.split(":")[1]
				stmtinfo.append(int(y.split(":")[0].split()[0]))
				stmtinfo.append(int(f.strip()))
				exereal.append(stmtinfo)
       #print("mapping data: "+str(exereal))
       for x in range(len(exereal)):
               if faultylno==exereal[x][0]:
			reallno=exereal[x][1]
       worstrank=0
       for x in range(len(sorted_func)):
               if (reallno>=sorted_func[x][2] and reallno<=sorted_func[x][3]):
			tstmt=0
			for z in range(0, int(sorted_func[x][6])-1):
				#print("index value of previous ranks z: "+str(z)+"\n")
				for y in range(int(sorted_func[z][2]), int(sorted_func[z][3])+1):
					#print("for particular rank, real line no: "+str(y)+"\n")
					for k in range(len(exereal)):
						if y==exereal[k][1]:
							#print("real line no which is also executable line: "+str(y))
							tstmt=tstmt+1
			print("total statements: "+str(tstmt))
			print("Best Rank of faulty line --->  "+str(tstmt+1))
			if sorted_func[x][2]==sorted_func[x][3]:
				print("Worst Rank of faulty line --->  "+str(tstmt+1))
			else:
				count=0
				for i in range(sorted_func[x][2], sorted_func[x][3]+1):
					for k in range(len(exereal)):
						if i==exereal[k][1]:
							#print("real line no which is also executable line for count: "+str(i))
							count=count+1
				worstrank=tstmt+count
				print("Worst Rank of faulty line --->  "+str(worstrank))

