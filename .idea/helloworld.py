from sklearn import tree

features = [[140, 1],[130, 1], [150, 0], [170,0]]
labels = [0,0,1,1]
#type of classifier to start with is called a decision tryy
#consider as box of rules
clf = tree.DecisionTreeClassifier()
#creates rule, heavier fruit is, more likely to be an orange
clf = clf.fit(features, labels)
print(clf.predict([[150,0]]))