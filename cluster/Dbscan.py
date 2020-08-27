import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

mac2id=dict()
onlinetimes=[]
f = open("TestData.txt",encoding='utf8')
for line in f:
    vals = line.split(",")
    mac = vals[2]
    onlinetime = int(vals[6])
    starttime = int(vals[4].split(' ')[1].split(':')[0])
    if mac not in mac2id:
        mac2id[mac]=len(onlinetimes)   #mac2id 存放mac地址的上网时长及开始上网时间onlinetimes的下标 key为mac地址
        onlinetimes.append((starttime,onlinetime))
    else:
        onlinetimes[mac2id[mac]]=[(starttime,onlinetime)]

real_X=np.array(onlinetimes).reshape((-1,2)) #行自动 2列
#print(real_X)
x = real_X[:,0:1] #上网时间
#print(x)

db = skc.DBSCAN(eps=0.01, min_samples=20).fit(x)
labels = db.labels_
print("Labels:")
print(labels)

ratio = len(labels[labels[:]==-1]) / len(labels)
print('noise ratio:',format(ratio,'.2%')) #噪声数据比例

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print('estimated numberof clusters:%d' % n_clusters)
print('silhouette coefficient:%0.3f' % metrics.silhouette_score(x ,labels)) #评价聚类效果

for i in range(n_clusters):
    print('cluters ',i,':')
    print(list(x[labels==i].flatten())) #打印各簇标号以及簇内数据

#plt.hist(x,24)
#plt.show()

X = np.log(1+real_X[:,1:]) #上网时长
print(X)