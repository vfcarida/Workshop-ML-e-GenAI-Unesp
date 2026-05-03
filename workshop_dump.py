# CELL 0
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid', {"axes.grid" : False})
sns.set_context('notebook')
np.random.seed(42)

# CELL 1
from sklearn.linear_model import LinearRegression

# CELL 2
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

# CELL 3
baseDados = baseDados = pd.read_csv('./base_regressao_1.csv')
LR = LinearRegression()
X = baseDados[['X']]
Y = baseDados.Y
LR.fit(X,Y)

plt.plot(baseDados.X, LR.predict(baseDados[['X']]), c = 'darkgreen')
plt.scatter(baseDados.X, baseDados.Y, s = 5, c = 'black')

# CELL 4
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
%matplotlib inline

# CELL 5
pesos = pd.read_csv('./weights.csv', sep=';')

# CELL 6
pesos.head()

# CELL 7
pesos.loc[pesos.sex=='F']['weight'].max()

# CELL 8
pesos.hist()

# CELL 9
pesos.dtypes

# CELL 10
pesos.shape

# CELL 11
pesos.plot()

# CELL 12
pesos.describe()

# CELL 13
pesos.sex.value_counts()

# CELL 14
plt.scatter(x=pesos.loc[pesos.sex=='M', 'height'],
            y=pesos.loc[pesos.sex=='M', 'weight'], c='blue')
plt.scatter(x=pesos.loc[pesos.sex=='F', 'height'],
            y=pesos.loc[pesos.sex=='F', 'weight'], c='red')

# CELL 15
pesos = pesos[pesos.weight <= 100]
plt.scatter(x=pesos.loc[pesos.sex=='M', 'height'],
            y=pesos.loc[pesos.sex=='M', 'weight'], c='blue')
plt.scatter(x=pesos.loc[pesos.sex=='F', 'height'],
            y=pesos.loc[pesos.sex=='F', 'weight'], c='red')

# CELL 16
pesos.head()

# CELL 17
dummies_sex = pd.get_dummies(pesos.sex, drop_first=True)
dummies_sex.head()

# CELL 18
pesos = pd.concat([pesos, dummies_sex], axis=1)
pesos.head()

# CELL 19
X = pesos[['M', 'height']]
y = pesos.weight

# CELL 20
#Teste
#X = pesos[['weight', 'height']]
#y = pesos.sex

# CELL 21
X.head()

# CELL 22
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# CELL 23
lin_reg.coef_

# CELL 24
lin_reg.intercept_

# CELL 25
heights = np.linspace(140, 210, 70)
sex_masc = np.ones(70)
sex_fem = np.zeros(70)

coef_sex = float(lin_reg.coef_[0])
coef_height = float(lin_reg.coef_[1])
intercept = float(lin_reg.intercept_)

reta_masc = coef_sex*sex_masc + coef_height*heights + intercept
reta_fem = coef_sex*sex_fem + coef_height*heights + intercept

# CELL 26
plt.scatter(x=pesos.loc[pesos.sex=='M', 'height'],
            y=pesos.loc[pesos.sex=='M', 'weight'], c='blue')
plt.scatter(x=pesos.loc[pesos.sex=='F', 'height'],
            y=pesos.loc[pesos.sex=='F', 'weight'], c='red')
plt.plot(heights, reta_masc, '-', c='blue')
plt.plot(heights, reta_fem, '-', c='red')
plt.show()

# CELL 27
lin_reg.predict([[0, 157]])

# CELL 28
lin_reg.predict([[1, 173]])

# CELL 29
from sklearn.metrics import mean_squared_error

# CELL 30
y_pred = lin_reg.predict(X)

# CELL 31
mean_squared_error(y, y_pred)

# CELL 32
a = lin_reg.predict([[0, 160]])
a

# CELL 33
b = lin_reg.predict([[1, 160]])
b

# CELL 34
c = b-a
c

# CELL 35
lin_reg.predict([[0, 60]])

# CELL 36
data = pd.read_csv('./boston.csv', sep=';', decimal=',')

# CELL 37
data.head()

# CELL 38
data.describe()

# CELL 39
data.hist(bins=50, figsize=(20,15))

# CELL 40
y = data['MV']
X = data.drop('MV', axis=1)

# CELL 41
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# CELL 42
x_new = [[0.02731, 0.0, 7.07, 0, 0.469, 6.421,
          78.900002, 4.9671, 2, 242, 17.799999, 396.899994, 9.14]]

# CELL 43
lin_reg.predict(x_new)

# CELL 44
X_new = [[0.02731, 0.0, 7.07, 0, 0.469, 6.421, 78.900002, 4.9671, 2,
          242, 17.799999, 396.899994, 9.14],
         [0.67671, 0.8, 5.56, 0, 0.567, 3.132, 60.678976, 2.3465, 3,
          432,  9.546666, 342.435664, 3.23],
         [0.05641, 0.0, 1.04, 1, 0.232, 4.322, 67.564646, 1.5678, 5,
          567, 15.898006, 224.655678, 5.84]]

# CELL 45
lin_reg.predict(X_new)

# CELL 46
lin_reg.coef_

# CELL 47
lin_reg.intercept_

# CELL 48
mean_squared_error(lin_reg.predict(X), y)

# CELL 49
pesos.head()

# CELL 50
#antes
#X = pesos[['M', 'height']]
#y = pesos.weight
#Agora
X1 = pesos[['height', 'weight']]
y1 = pesos.M
#y1 = pesos.sex

# CELL 51
lin_reg1 = LinearRegression()
lin_reg1.fit(X1, y1)

# CELL 52
lin_reg1.predict([[173, 80]])

# CELL 53
y_pred1 = lin_reg1.predict(X1)

# CELL 54
mean_squared_error(y1, y_pred1)

# CELL 55
from sklearn.linear_model import LogisticRegression
reg_log = LogisticRegression(C=1)
reg_log.fit(X1, y1)

# CELL 56
reg_log.predict([[173, 80]])

# CELL 57
y_pred2 = reg_log.predict(X1)

# CELL 58
from sklearn.metrics import accuracy_score

# CELL 59
accuracy_score(y1, y_pred2)

# CELL 60
accuracy_score(y1, y_pred2, normalize=False)

# CELL 61
from sklearn.tree import DecisionTreeClassifier
arvore = DecisionTreeClassifier(criterion="entropy", max_depth=2)
arvore.fit(X1, y1)

# CELL 62
arvore.predict([[173, 80]])

# CELL 63
y_pred3 = arvore.predict(X1)

# CELL 64
accuracy_score(y1, y_pred3)

# CELL 65
accuracy_score(y1, y_pred3, normalize=False)

# CELL 66
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y1, y_pred3)
confusion_matrix

# CELL 67
from sklearn.metrics import classification_report
print(classification_report(y1, y_pred3))

# CELL 68
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid', {"axes.grid" : False})
sns.set_context('notebook')
np.random.seed(42)

# CELL 69
baseDados = pd.read_csv('./base_regressao_logistica.csv')
plt.scatter(baseDados.X1[baseDados.Y == 0], baseDados.X2[baseDados.Y == 0], c = 'darkgreen', marker = '^')
plt.scatter(baseDados.X1[baseDados.Y == 1], baseDados.X2[baseDados.Y == 1], c = 'black', marker = ',')

# CELL 70
from sklearn.linear_model import LogisticRegression

# CELL 71
LR = LogisticRegression(C = 1)
X = baseDados.loc[:, baseDados.columns != 'Y']
Y = baseDados.Y
LR.fit(X,Y)

# CELL 72
x_min, x_max = baseDados.X1.min() - .1, baseDados.X1.max() + .1
y_min, y_max = baseDados.X2.min() - .1, baseDados.X2.max() + .1
h = .005
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = LR.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = plt.cm.Accent)

pred = LR.predict(X)
plt.scatter(baseDados.X1[Y == 0], baseDados.X2[Y == 0], c = 'darkgreen', marker = '^')
plt.scatter(baseDados.X1[Y == 1], baseDados.X2[Y == 1], c = 'black', marker = ',')

# CELL 73
baseDados = pd.read_csv('./base_regressao_logistica.csv')
plt.scatter(baseDados.X1[baseDados.Y == 0], baseDados.X2[baseDados.Y == 0], c = 'darkgreen', marker = '^')
plt.scatter(baseDados.X1[baseDados.Y == 1], baseDados.X2[baseDados.Y == 1], c = 'black', marker = ',')

# CELL 74
from sklearn.tree import DecisionTreeClassifier

# CELL 75
DT = DecisionTreeClassifier(criterion="entropy", max_depth=2)
X = baseDados.loc[:, baseDados.columns != 'Y']
Y = baseDados.Y
DT.fit(X,Y)

# CELL 76
x_min, x_max = baseDados.X1.min() - .1, baseDados.X1.max() + .1
y_min, y_max = baseDados.X2.min() - .1, baseDados.X2.max() + .1
h = .005
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = DT.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = plt.cm.Accent)

pred = DT.predict(X)
plt.scatter(baseDados.X1[Y == 0], baseDados.X2[Y == 0], c = 'darkgreen', marker = '^')
plt.scatter(baseDados.X1[Y == 1], baseDados.X2[Y == 1], c = 'black', marker = ',')

# CELL 77
from sklearn.neighbors import KNeighborsClassifier

# CELL 78
baseDados = pd.read_csv('./base_knn.csv')

h = .01
x_min, x_max = baseDados.X1.min() - 1, baseDados.X1.max() + 1
y_min, y_max = baseDados.X2.min() - 1, baseDados.X2.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plt.scatter(baseDados.X1[baseDados.Y == 0], baseDados.X2[baseDados.Y == 0], c = 'darkgreen', marker = '_')
plt.scatter(baseDados.X1[baseDados.Y == 1], baseDados.X2[baseDados.Y == 1], c = 'black', marker = '+')

# CELL 79
KNN = KNeighborsClassifier(n_neighbors = 2)
X = baseDados[['X1','X2']]
Y = baseDados.Y
KNN.fit(X,Y)

Z = KNN.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = plt.cm.Accent)

pred = KNN.predict(X)
plt.scatter(baseDados.X1[Y == 0], baseDados.X2[Y == 0], c = 'darkgreen', marker = '_')
plt.scatter(baseDados.X1[Y == 1], baseDados.X2[Y == 1], c = 'black', marker = '+')
plt.scatter([2.5],[2.5], s = 100, c = 'darkgreen' if KNN.predict([[2.5,2.5]]) == 0 else 'black')
plt.scatter([1.5],[1.0], s = 100, c = 'darkgreen' if KNN.predict([[1.5,1.0]]) == 0 else 'black')
plt.scatter([3.5],[3.0], s = 100, c = 'darkgreen' if KNN.predict([[3.5,3.0]]) == 0 else 'black')

# CELL 80
from sklearn.naive_bayes import GaussianNB

# CELL 81
baseDados = pd.read_csv('./base_regressao_logistica.csv')
plt.scatter(baseDados.X1[baseDados.Y == 0], baseDados.X2[baseDados.Y == 0], c = 'darkgreen', marker = '^')
plt.scatter(baseDados.X1[baseDados.Y == 1], baseDados.X2[baseDados.Y == 1], c = 'black', marker = ',')

# CELL 82
NB = GaussianNB()
X = baseDados.loc[:, baseDados.columns != 'Y']
Y = baseDados.Y
NB.fit(X,Y)

# CELL 83
x_min, x_max = baseDados.X1.min() - .1, baseDados.X1.max() + .1
y_min, y_max = baseDados.X2.min() - .1, baseDados.X2.max() + .1
h = .005
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = NB.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = plt.cm.Accent)

pred = NB.predict(X)
plt.scatter(baseDados.X1[Y == 0], baseDados.X2[Y == 0], c = 'darkgreen', marker = '^')
plt.scatter(baseDados.X1[Y == 1], baseDados.X2[Y == 1], c = 'black', marker = ',')

# CELL 84
#http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

# CELL 85
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# CELL 86
h = .02  # step size in the mesh

# CELL 87
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

# CELL 88
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# CELL 89
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

# CELL 90
figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()

# CELL 91
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid', {"axes.grid" : False})
sns.set_context('notebook')
np.random.seed(42)

# CELL 92
from scipy.cluster import hierarchy

# CELL 93
baseDados = pd.read_csv('./base_agrupamento_2.csv')
X = baseDados[['X','Y']]

# CELL 94
Z = hierarchy.linkage(X, method = 'single')
dendro = hierarchy.dendrogram(Z)
plt.show()

# CELL 95
Z = hierarchy.linkage(X, method = 'complete')
dendro = hierarchy.dendrogram(Z)
plt.show()

# CELL 96
Z = hierarchy.linkage(X, method = 'single')
baseDados['hierarquico'] = hierarchy.fcluster(Z, 2, criterion = 'maxclust') - 1

markers = {0 : '^', 1 : 'x'}
for clu in baseDados.hierarquico.unique():
    plt.scatter(baseDados.X[baseDados.hierarquico == clu],
                baseDados.Y[baseDados.hierarquico == clu],
                s = 50,
                marker = markers[clu],
                c = plt.cm.Accent.colors[clu])

# CELL 97
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid', {"axes.grid" : False})
sns.set_context('notebook')
np.random.seed(42)

# CELL 98
baseDados = pd.read_csv('./base_agrupamento_1.csv')

markers = {1 : '+', 2 : 'x', 3 : '^'}
for clu in baseDados.clu.unique():
    plt.scatter(baseDados.X[baseDados.clu == clu],
                baseDados.Y[baseDados.clu == clu],
                s = 50,
                marker = markers[clu],
                c = plt.cm.Accent.colors[clu - 1])

# CELL 99
baseDados = pd.read_csv('./base_agrupamento_2.csv')

markers = {1 : '^', 2 : 'x'}
for clu in baseDados.clu.unique():
    plt.scatter(baseDados.X[baseDados.clu == clu],
                baseDados.Y[baseDados.clu == clu],
                s = 50,
                marker = markers[clu],
                c = plt.cm.Accent.colors[clu - 1])

# CELL 100
baseDados = pd.read_csv('./iris.csv')

markers = {'setosa' : '+', 'virginica' : 'x', 'versicolor' : '^'}
colors = {'setosa' : 0, 'virginica' : 1, 'versicolor' : 2}
for clu in baseDados.species.unique():
    plt.scatter(baseDados['sepal length (cm)'][baseDados.species == clu],
                baseDados['petal length (cm)'][baseDados.species == clu],
                s = 50,
                marker = markers[clu],
                c = plt.cm.Accent.colors[colors[clu]])

# CELL 101
from sklearn.cluster import KMeans

# CELL 102
np.random.seed(42)
baseDados = pd.read_csv('./base_agrupamento_1.csv')
X = baseDados[['X','Y']]
centers = np.array([[np.random.uniform(baseDados.X.min(), baseDados.X.max()),
            np.random.uniform(baseDados.Y.min(), baseDados.Y.max())] for i in range(3)])

h = .005
x_min, x_max = baseDados.X.min() - 1, baseDados.X.max() + 1
y_min, y_max = baseDados.Y.min() - 1, baseDados.Y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# CELL 103
for _ in range(10):
    km = KMeans(n_clusters = 3, n_init = 1, init = centers, max_iter=1)
    baseDados['kmeans'] = km.fit_predict(X)

    # Nesta parte imprimimos a area colorida no fundo
    Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Accent,
               aspect='auto', origin='lower')

    # Nesta parte imprimimos os pontos
    markers = {0 : '+', 1 : 'x', 2 : '^'}
    for clu in baseDados.kmeans.unique():
        plt.scatter(baseDados.X[baseDados.kmeans == clu],
                    baseDados.Y[baseDados.kmeans == clu],
                    s = 50,
                    marker = markers[clu],
                    c = 'black')

    for centroide in km.cluster_centers_:
        plt.scatter(centroide[0], centroide[1], s = 250, c = 'white')
    centers = km.cluster_centers_
    plt.show()

# CELL 104
km = KMeans(n_clusters = 3, n_init = 3, init = 'k-means++', max_iter=300)
baseDados['kmeans'] = km.fit_predict(X)

# Nesta parte imprimimos a area colorida no fundo
Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Accent,
           aspect='auto', origin='lower')

# Nesta parte imprimimos os pontos
markers = {0 : '+', 1 : 'x', 2 : '^'}
for clu in baseDados.kmeans.unique():
    plt.scatter(baseDados.X[baseDados.kmeans == clu],
                baseDados.Y[baseDados.kmeans == clu],
                s = 50,
                marker = markers[clu],
                c = 'black')

for centroide in km.cluster_centers_:
    plt.scatter(centroide[0], centroide[1], s = 250, c = 'white')
centers = km.cluster_centers_
plt.show()

# CELL 105
baseDados = pd.read_csv('./base_agrupamento_2.csv')
X = baseDados[['X','Y']]

h = .005
x_min, x_max = baseDados.X.min() - 1, baseDados.X.max() + 1
y_min, y_max = baseDados.Y.min() - 1, baseDados.Y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# CELL 106
km = KMeans(n_clusters = 2, n_init = 3, init = 'k-means++', max_iter=300)
baseDados['kmeans'] = km.fit_predict(X)

# Nesta parte imprimimos a area colorida no fundo
Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Accent,
           aspect='auto', origin='lower')

# Nesta parte imprimimos os pontos
markers = {0 : '+', 1 : 'x', 2 : '^'}
for clu in baseDados.kmeans.unique():
    plt.scatter(baseDados.X[baseDados.kmeans == clu],
                baseDados.Y[baseDados.kmeans == clu],
                s = 50,
                marker = markers[clu],
                c = 'black')

for centroide in km.cluster_centers_:
    plt.scatter(centroide[0], centroide[1], s = 250, c = 'white')
centers = km.cluster_centers_
plt.show()

# CELL 107
km = KMeans(n_clusters = 3, n_init = 10, init = 'k-means++', max_iter = 300)

# CELL 108
km.fit(X)

# CELL 109
baseDados['kmeans'] = km.predict(X)
baseDados.head()

# CELL 110
km.cluster_centers_

# CELL 111
# Centroide do cluster k
k = 0
km.cluster_centers_[k, :]

# CELL 112


# CELL 113
%matplotlib inline

# CELL 114
print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# CELL 115


# CELL 116
print(__doc__)

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt


from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}

datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2,
                     'min_samples': 20, 'xi': 0.25}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2,
              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
    (aniso, {'eps': .15, 'n_neighbors': 2,
             'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
    (blobs, {}),
    (no_structure, {})]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    #optics = cluster.OPTICS(min_samples=params['min_samples'],
    #                        xi=params['xi'],
    #                        min_cluster_size=params['min_cluster_size'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('AffinityPropagation', affinity_propagation),
        ('MeanShift', ms),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        #('OPTICS', optics),
        ('Birch', birch),
        ('GaussianMixture', gmm)
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int_)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()

# CELL 117


# CELL 118
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)
print(os.listdir("./"))

# CELL 119
# Importing the dataset
dataset = pd.read_csv('./Mall_Customers.csv',index_col='CustomerID')

# CELL 120
dataset.head()

# CELL 121
dataset.info()

# CELL 122
dataset.describe()

# CELL 123
dataset.isnull().sum()

# CELL 124
dataset.drop_duplicates(inplace=True)

# CELL 125
# using only Spending_Score and income variable for easy visualisation
X = dataset.iloc[:, [2, 3]].values

# CELL 126
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

# CELL 127
plt.figure(figsize=(10,5))
sns.lineplot(x=range(1, 11), y=wcss,marker='o',color='red') # Pass x and y as keyword arguments.
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# CELL 128
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# CELL 129
# Visualising the clusters
plt.figure(figsize=(15,7))
sns.scatterplot(x=X[y_kmeans == 0, 0], y=X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1',s=50) # Pass x and y as keyword arguments.
sns.scatterplot(x=X[y_kmeans == 1, 0], y=X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2',s=50) # Pass x and y as keyword arguments.
sns.scatterplot(x=X[y_kmeans == 2, 0], y=X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3',s=50) # Pass x and y as keyword arguments.
sns.scatterplot(x=X[y_kmeans == 3, 0], y=X[y_kmeans == 3, 1], color = 'grey', label = 'Cluster 4',s=50) # Pass x and y as keyword arguments.
sns.scatterplot(x=X[y_kmeans == 4, 0], y=X[y_kmeans == 4, 1], color = 'orange', label = 'Cluster 5',s=50) # Pass x and y as keyword arguments.
sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], color = 'red', # Pass x and y as keyword arguments.
                label = 'Centroids',s=300,marker=',')
plt.grid(False)
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# CELL 130
! pip3 install --upgrade --user google-cloud-aiplatform pymupdf

# CELL 131
!pip3 install tensorflow-text

# CELL 132
!pip3 install cohere umap-learn altair annoy datasets tqdm

# CELL 133
!pip3 install --upgrade cohere

# CELL 134
!pip install diffusers==0.11.1
!pip install transformers scipy ftfy accelerate

# CELL 135
!pip install "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CELL 136
import IPython
import time

app = IPython.Application.instance()
app.kernel.do_shutdown(True)

# CELL 137
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()

# CELL 138
# Define project information

PROJECT_ID = "vinicaridateste"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# if not running on colab, try to get the PROJECT_ID automatically
if "google.colab" not in sys.modules:
    import subprocess

    PROJECT_ID = subprocess.check_output(
        ["gcloud", "config", "get-value", "project"], text=True
    ).strip()

print(f"Your project ID is: {PROJECT_ID}")

# CELL 139
import sys

if "google.colab" in sys.modules:
    # Initialize Vertex AI
    import vertexai

    vertexai.init(project=PROJECT_ID, location=LOCATION)

# CELL 140
import cohere
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import umap
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

# CELL 141
from IPython.display import Markdown, display
from vertexai.preview.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Image,
    Part,
)
from vertexai.language_models import TextEmbeddingModel

# CELL 142
import getpass
api_key = getpass.getpass("enter your co.here api token")

# CELL 143
from vertexai.preview.generative_models import GenerativeModel, Image

# CELL 144
!nvidia-smi

# CELL 145
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

# CELL 146
pipe = pipe.to("cuda")

# CELL 147
multimodal_model = GenerativeModel("gemini-pro-vision")

# CELL 148
import http.client
import io
import typing
import urllib.request

import IPython.display
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps

def display_image(image: Image,
                  max_width: int = 600,
                  max_height: int = 350) -> None:
    pil_image = typing.cast(PIL_Image.Image, image._pil_image)
    if pil_image.mode != "RGB":
        # Modes such as RGBA are not yet supported by all Jupyter environments
        pil_image = pil_image.convert("RGB")
    image_width, image_height = pil_image.size
    if max_width < image_width or max_height < image_height:
        # Resize to display a smaller notebook image
        pil_image = PIL_ImageOps.contain(pil_image, (max_width, max_height))
    display_image_compressed(pil_image)


def display_image_compressed(pil_image: PIL_Image.Image) -> None:
    image_io = io.BytesIO()
    pil_image.save(image_io, "jpeg", quality=80, optimize=True)
    image_bytes = image_io.getvalue()
    ipython_image = IPython.display.Image(image_bytes)
    IPython.display.display(ipython_image)


def get_image_bytes_from_url(image_url: str) -> bytes:
    with urllib.request.urlopen(image_url) as response:
        response = typing.cast(http.client.HTTPResponse, response)
        if response.headers["Content-Type"] not in ("image/png", "image/jpeg"):
            raise Exception("Image can only be in PNG or JPEG format")
        image_bytes = response.read()
    return image_bytes


def load_image_from_url(image_url: str) -> Image:
    image_bytes = get_image_bytes_from_url(image_url)
    return Image.from_bytes(image_bytes)


def print_multimodal_prompt(contents: list):
    """
    Given contents that would be sent to Gemini,
    output the full multimodal prompt for ease of readability.
    """
    for content in contents:
        if isinstance(content, Image):
            display_image(content)
        else:
            print(content)

# CELL 149
# urls for room images
room_image_url = "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/rooms/spacejoy-c0JoR_-2x3E-unsplash.jpg"

# load room images as Image Objects
room_image = load_image_from_url(room_image_url)

prompt = "Descreva o que é visível nesta sala e sua atmosfera geral:"
contents = [
    prompt,
    room_image,
]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Resposta--------")
for response in responses:
    print(response.text, end="")

# CELL 150
prompt1 = "Recomende um novo móvel para esta sala:"
prompt2 = "e explique a razão em detalhes"
contents = [
    prompt1,
    room_image,
    prompt2
]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Resposta--------")
for response in responses:
    print(response.text, end="")

# CELL 151
prompt1 = "Descreva esta sala:"
prompt2 = "e recomende um tipo de tapete que caiba nele. Um tapete grande e neutro com padrão geométrico ajudaria a definir o espaço e adicionaria um toque de aconchego."
contents = [
    prompt1,
    room_image,
    prompt2
]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")

# CELL 152
prompt1 = "Descreva esta sala:"
prompt2 = "e recomende um tipo de cadeira que combine com ela"
contents = [
    prompt1,
    room_image,
    prompt2
]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Resposta--------")
for response in responses:
    print(response.text, end="")

# CELL 153
#from PIL import Image
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL_Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# CELL 154
num_images = 3
prompt = ["Generate a photo of a wooden chair with a simple and modern design. The chair could have a seat and back upholstered in a light-colored fabric. The chair could also have wooden arms and legs."] * num_images

images = pipe(prompt).images

grid = image_grid(images, rows=1, cols=3)
grid

# CELL 155
# Download and display sample chairs
furniture_image_urls = [
    "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/cesar-couto-OB2F6CsMva8-unsplash.jpg",
    "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/daniil-silantev-1P6AnKDw6S8-unsplash.jpg",
    "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/ruslan-bardash-4kTbAMRAHtQ-unsplash.jpg",
    "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/scopic-ltd-NLlWwR4d3qU-unsplash.jpg",
]

# Load furniture images as Image Objects
furniture_images = [load_image_from_url(url) for url in furniture_image_urls]

# To recommend an item from a selection, you will need to label the item number within the prompt.
# That way you are providing the model with a way to reference each image as you pose a question.
# Labelling images within your prompt also help to reduce hallucinations and overall produce better results.
contents = [
    "Considere as seguintes cadeiras:",
    "cadeira 1:", furniture_images[0],
    "cadeira 2:", furniture_images[1],
    "cadeira 3:", furniture_images[2],
    "cadeira 4:", furniture_images[3],
    "sala:",
    room_image,
    "Você é um designer de interiores. Para cada cadeira, explique se ela seria apropriada ou não para o estilo da sala. Sumarize quais as duas melhores opções de cadeira:",
]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Resposta--------")
for response in responses:
    print(response.text, end="")

# CELL 156
contents = [
    "Considere as seguintes cadeiras:",
    "cadeira 1:", furniture_images[0],
    "cadeira 2:", furniture_images[1],
    "cadeira 3:", furniture_images[2],
    "cadeira 4:", furniture_images[3],
    "sala:",
    room_image,
    "Você é um designer de interiores. Retorne em JSON, para cada cadeira, se ela seria uma boa escolha ou não para esta sala, incluindo uma explicação da decisão:",
]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Resposta--------")
for response in responses:
    print(response.text, end="")

