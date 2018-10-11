from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets


irisdataset = datasets.load_iris()
x = irisdataset.data
y = irisdataset.target

model = KNeighborsClassifier(n_neighbors=2)
model.fit(x, y)
print("k = 2")
print(model.predict([[4, 3, 2, 1], [1, 2, 3, 4], [2, 3, 1, 1]]))
print(model.score(x, y))

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)
print("k = 3")
print(model.predict([[4, 3, 2, 1], [1, 2, 3, 4], [2, 3, 1, 1]]))
print(model.score(x, y))

model = KNeighborsClassifier(n_neighbors=4)
model.fit(x, y)
print("k = 4")
print(model.predict([[4, 3, 2, 1], [1, 2, 3, 4], [2, 3, 1, 1]]))
print(model.score(x, y))

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x, y)
print("k = 4")
print(model.predict([[4, 3, 2, 1], [1, 2, 3, 4], [2, 3, 1, 1]]))
print(model.score(x, y))

model = KNeighborsClassifier(n_neighbors=6)
model.fit(x, y)
print("k = 6")
print(model.predict([[4, 3, 2, 1], [1, 2, 3, 4], [2, 3, 1, 1]]))
print(model.score(x, y))

model = KNeighborsClassifier(n_neighbors=8)
model.fit(x, y)
print("k = 8")
print(model.predict([[4, 3, 2, 1], [1, 2, 3, 4], [2, 3, 1, 1]]))
print(model.score(x, y))

model = KNeighborsClassifier(n_neighbors=8)
model.fit(x, y)
print("k = 10")
print(model.predict([[4, 3, 2, 1], [1, 2, 3, 4], [2, 3, 1, 1]]))
print(model.score(x, y))