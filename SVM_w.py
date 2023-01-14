# %% [markdown]
# # 1. Aquisição e transformação dos dados

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from imblearn.metrics import geometric_mean_score

# %%
w_wine = pd.read_csv('datasets/winequality-white.csv', sep = ';')

# %%
x = w_wine.iloc[:, 0:-1]
y = w_wine.iloc[:, -1]

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
x_train = np.array(x_train)
x_test = np.array(x_test)

# %%
counter = Counter(y_train)
plt.bar(counter.keys(), counter.values())
plt.savefig('originaçl_data_w.png')
plt.show()

# %% [markdown]
# # 2. ADASYN e Undersampling

# %%
wg = y_train.value_counts()

# %%
for i in range(0, len(wg)):
    if wg[i+3] < 6:
        wg[i+3] = 6

strategy1_w = {3:wg[3], 4:wg[4], 5:wg[5], 6:wg[6], 7:wg[7], 8:wg[8], 9:wg[9]}
fix = RandomOverSampler(sampling_strategy=strategy1_w)

# %%
wg[8] = round(0.2*wg[6])
wg[4] = round(0.2*wg[6])
wg[9] = round(0.1*wg[6])
wg[3] = round(0.15*wg[6])

strategy2_w = {3:wg[3], 4:wg[4], 8:wg[8], 9:wg[9]}
over = ADASYN(sampling_strategy=strategy2_w)

# %%
wg[5] = round(0.7*wg[5])
wg[6] = round(0.5*wg[6])

strategy3_w = {5:wg[5], 6:wg[6]}
under = RandomUnderSampler(sampling_strategy=strategy3_w)

# %%
steps = [('f', fix), ('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
x_train, y_train = pipeline.fit_resample(x_train, y_train)

# %%
counter = Counter(y_train)
plt.bar(counter.keys(), counter.values())
plt.savefig('SMOTE_data_w.png')
plt.show()

# %% [markdown]
# # 3. Implementação da RNA

# %% [markdown]
# ## 3.1 Descoberta dos hyperparâmetros (Otimização Bayseana)

# %%
# Import packages
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, KFold
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, recall_score, f1_score
from bayes_opt import BayesianOptimization
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
from scikeras.wrappers import KerasClassifier
from sklearn import metrics
import math

pd.set_option("display.max_columns", None)

# %%
score_acc = make_scorer(roc_auc_score)

# %%
x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = scale(x_train)
x_test = scale(x_test)

# %%
y_train.shape

# %%
def svm(c, gamma, w3, w4, w5, w6, w7, w8, w9):

    weights = [w3, w4, w5, w6, w7, w8, w9]

    w = {}
    for i in range(0, len(weights)):
        index = i+3
        w[index] = weights[i]

    svm_w = SVC(random_state=0, C=c, gamma=gamma, class_weight= w, probability=True)

    return svm_w

# %%
def evaluate_model(c, gamma, w3, w4, w5, w6, w7, w8, w9):
    
    svm_w = svm(c, gamma, w3, w4, w5, w6, w7, w8, w9)

    kf = KFold(n_splits=10, random_state=0, shuffle=True)
    score = []
    
    for train_index, test_index in kf.split(x_train):
        # print("TRAIN:", train_index, "TEST:", test_index)
        xtr, xte = x_train[train_index], x_train[test_index]
        ytr, yte = y_train[train_index], y_train[test_index]

        svm_w.fit(xtr, ytr)
        y_pred = svm_w.predict(xte)
        proba = svm_w.predict_proba(xte)

        roc = roc_auc_score(yte, proba, multi_class='ovr')
        acc = accuracy_score(yte, y_pred)
        score.append(math.sqrt(acc*roc))

    score = sum(score)/len(score)
    return score

# %%
pbounds = {'c': (0,100),
            'gamma': (0,100),
            'w3': (0,100),
            'w4': (0,100),
            'w5': (0,100),
            'w6': (0,100),
            'w7': (0,100),
            'w8': (0,100),
            'w9': (0,100),
            }

optimizer = BayesianOptimization(
    f=evaluate_model,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum 
    # is observed, verbose = 0 is silent
    random_state=1
)

# %%
import warnings
warnings.filterwarnings('ignore')
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

optimizer.maximize(init_points=250, n_iter=200,)

# %%
params_ = optimizer.max['params']
params_

# %%
svm_w = svm(params_['c'], params_['gamma'], params_["w3"], params_["w4"], params_["w5"], 
            params_["w6"], params_["w7"], params_["w8"], params_["w9"])


svm_w.fit(x_train, y_train)

y_pred = svm_w.predict(x_test)

# %%
svm_w.score(x_test, y_test)

# %%
cm = metrics.confusion_matrix(y_test, y_pred)
cm

# %%
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [3,4,5,6,7,8,9])
cm_display.plot()

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(15,8))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[3,4,5,6,7,8,9], yticklabels=[3,4,5,6,7,8,9])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


