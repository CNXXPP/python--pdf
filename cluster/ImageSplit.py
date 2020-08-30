import numpy as np
import PIL.Image as image #加载PIL包，用于加载创建图片
from sklearn.cluster import KMeans #加载Kmeans算法

def loadData(filePath):
    f = open(filePath,'rb')
    data = []
    img = image.open(f)
    m,n=img.size
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    return np.mat(data),m,n

imgData,m,n=loadData('1.jpg')

km = KMeans(n_clusters=6)

label = km.fit_predict(imgData)
label = label.reshape([m,n])
pic_new=image.new("L",(m,n))
for i in range(m):
    for j in range(n):
        pic_new.putpixel((i, j), int(256/(label[i][j]+1)))

pic_new.save("result1.jpg","JPEG")