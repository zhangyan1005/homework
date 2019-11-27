import numpy as np
from PIL import Image
#from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

people = []
people_target = []

for j in range(1,41):
    path1 = "s" + str(j) + "/"
    for i in range(1,11):
        path = "./image/" +  path1 + str(i) + ".pgm"
        im = Image.open(path)    # 读取文件
        temp = np.array(im)
        im = temp.flatten()
        people.append(im)
        people_target.append(j)
        
people = np.array(people)
people_target = np.array(people_target)
#im.show()    # 展示图片
X_people = people
y_people = people_target
# 数据训练测试集 分割
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, 
                                            stratify=y_people)# random_state=0
lda = LinearDiscriminantAnalysis(n_components=39)
lda.fit(X_train,y_train) # lda模型训练
X_train_lda = lda.transform(X_train) # 转换 
X_test_lda = lda.transform(X_test) # 转换
print("X_train_lda.shape: {}".format(X_train_lda.shape))

# 使用k临近分类器
knn = KNeighborsClassifier(n_neighbors=1)  
knn.fit(X_train_lda, y_train) 
knn.predict(X_test_lda)
print("使用LDA降维, Test set accuracy: {:.2f}".format(knn.score(X_test_lda, y_test)))
# 使用k临近分类器
knn = KNeighborsClassifier(n_neighbors=1)  
knn.fit(X_train, y_train) 
print("不使用LDA降维 :Test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))