import pandas as pd

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
dir = 'C:/Users/15/Desktop/DataSet/'
df = pd.read_csv(dir+"/[Dataset]_Module_18_(iris).data",header=None)
names = ["sepal_length", "sepal_width","petal_length", "petal_width", "class"]
df.columns = names
print(df.head())

# 각 클래스의 데이터 포인트 수 출력
print(df['class'].value_counts())

# 다른 클래스에 대해 다른 숫자를 지정하는 딕셔너리
label_encode = {"class": {"Iris-setosa":1}}

# .replace를 사용하여 다른 클래스를 숫자로 변경
df.replace(label_encode,inplace=True)

# 각 클래스의 데이터 포인트 수를 출력하여 클래스가 숫자로 변경되었는지 확인

df.head(150)

# 각 클래스의 데이터 포인트 수 출력
print(df['class'].value_counts())

# 다른 클래스에 대해 다른 숫자를 지정하는 딕셔너리
label_encode = {"class": {"Iris-setosa":1, "Iris-versicolor":2, "Iris-virginica":3}}

# .replace를 사용하여 다른 클래스를 숫자로 변경
df.replace(label_encode,inplace=True)

# 각 클래스의 데이터 포인트 수를 출력하여 클래스가 숫자로 변경되었는지 확인
print(df['class'].value_counts())

df2 = pd.read_csv(dir+"/[Dataset]_Module_18_(iris).data",header=None)
names = ["sepal_length", "sepal_width","petal_length", "petal_width", "class"]
df2.columns = names

df2 = pd.get_dummies(df2,prefix=['class'])
print(df2.head())

# KNeighborsClassifier 초기화
KNN = KNeighborsClassifier()

# x 값과 y 값을 추출합니다. x는 특성이고 y는 클래스입니다.
x = df[['sepal_length','sepal_width']]
y = df['class']

# 데이터가 올바른지 확인하기 위해 x 및 y의 .head()를 출력합니다.
print(x.head())
print(y.head())

# x 및 y 값을 사용하여 KNN을 훈련시킵니다. 이것은 .fit 메소드를 통해 수행됩니다.
knn = KNN.fit(x,y)

# sepal_length = 5 및 sepal_width = 3인 경우 학습된 KNN을 사용하여 꽃의 유형을 예측합니다.
# .predict 메서드를 사용할 수 있습니다.
test = pd.DataFrame()
test['sepal_length'] = [6]
test['sepal_width'] = [2.5]
predict_flower = KNN.predict(test)

# predict_flower 출력
print(predict_flower)

# KNeighborsClassifier 초기화
KNN = KNeighborsClassifier()

# x 값과 y 값을 추출합니다. x는 특성이고 y는 클래스입니다.
x = df[['sepal_length','sepal_width']]
y = df['class']

# 데이터가 올바른지 확인하기 위해 x 및 y의 .head()를 출력합니다.
print(x.head())
print(y.head())

# x 및 y 값을 사용하여 KNN을 훈련시킵니다. 이것은 .fit 메소드를 통해 수행됩니다.
knn = KNN.fit(x,y)

# sepal_length = 5 및 sepal_width = 3인 경우 학습된 KNN을 사용하여 꽃의 유형을 예측합니다.
# .predict 메서드를 사용할 수 있습니다.
test = pd.DataFrame()
test['sepal_length'] = [3]
test['sepal_width'] = [5]
predict_flower = KNN.predict(test)

# predict_flower 출력
print(predict_flower)

# KNeighborsClassifier 초기화
KNN2 = KNeighborsClassifier(n_neighbors=7)

# x 값과 y 값을 추출합니다. x는 특성이고 y는 클래스입니다.
x = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df['class']

# 데이터가 올바른지 확인하기 위해 x와 y의 .head()를 출력합니다.
print(x.head())
print(y.head())

# x 및 y 값을 사용하여 KNN을 훈련시킵니다. 이것은 .fit 메소드를 통해 수행됩니다.
KNN2 = KNN2.fit(x,y)

# sepal_length = 5.8, sepal_width = 2.3, feather_length = 5.0 및 feather_width = 1.3인 경우 훈련된 KNN2를 사용하여 꽃의 유형을 예측합니다.
# .predict 메서드를 사용할 수 있습니다.
test = pd.DataFrame()
test['sepal_length'] = [5.8]
test['sepal_width'] = [2.3]
test['petal_length'] = [5.0]
test['petal_width'] = [1.3]
predict_flower = KNN2.predict(test)

# predict_flower 출력
print(predict_flower)

# 의사 결정 트리 초기화
dt = tree.DecisionTreeClassifier()

# x 값과 y 값을 추출합니다. x는 sepal_length, sepal_width이고 y는 클래스입니다.
x = df[['sepal_length','sepal_width']]
y = df['class']

# 데이터가 올바른지 확인하기 위해 x 및 y의 .head()를 출력합니다.
print(x.head())
print(y.head())

# x 및 y 값을 사용하여 의사결정 트리를 훈련시킵니다. 이것은 .fit 메소드를 통해 수행됩니다.
dt = dt.fit(x,y)

# sepal_length 및 sepal_width를 열로 사용하여 test2라는 데이터 프레임을 만듭니다.

test2 = pd.DataFrame()
test2['sepal_length'] = [5]
test2['sepal_width'] = [3]

# .predict 메서드를 사용하여 새 꽃을 예측합니다. 예측된 꽃을 predict_flower라고 부를 수 있습니다.
predict_flower = dt.predict(test2)

# predict_flower 출력
print(predict_flower)