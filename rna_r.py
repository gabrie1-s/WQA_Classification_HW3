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
r_wine = pd.read_csv('datasets/winequality-red.csv', sep = ';')

# %%
x = r_wine.iloc[:, 0:-1]
y = r_wine.iloc[:, -1]

# %%
from sklearn.model_selection import train_test_split
# y = pd.get_dummies(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=111)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
counter = Counter(y_train)
plt.bar(counter.keys(), counter.values())
plt.show()

# %%
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# %% [markdown]
# # 2. ADASYN e Undersampling

# %%
wg = y_train.value_counts()
for i in range(0, len(wg)):
    if wg[i+3] < 6:
        wg[i+3] = 6

strategy1_w = {3:wg[3], 4:wg[4], 5:wg[5], 6:wg[6], 7:wg[7], 8:wg[8]}
fix = RandomOverSampler(sampling_strategy=strategy1_w)
wg[8] = round(0.15*wg[5])
wg[4] = round(0.2*wg[5])
wg[3] = round(0.1*wg[5])

strategy2_w = {3:wg[3], 4:wg[4], 8:wg[8]}
over = ADASYN(sampling_strategy=strategy2_w)
wg[5] = round(0.5*wg[5])
wg[6] = round(0.6*wg[6])

strategy3_w = {5:wg[5], 6:wg[6]}
under = RandomUnderSampler(sampling_strategy=strategy3_w)
steps = [('f', fix), ('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
x_train, y_train = pipeline.fit_resample(x_train, y_train)
counter = Counter(y_train)
plt.bar(counter.keys(), counter.values())
plt.show()

# %%
n_classes = pd.get_dummies(y).shape[1]
n_classes

# %% [markdown]
# # 3. Implementação da RNA

# %% [markdown]
# ## 3.1 Descoberta dos hyperparâmetros (Otimização Bayseana)

# %%
# Import packages
from sklearn.model_selection import cross_val_score, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import make_scorer, accuracy_score,  roc_auc_score, accuracy_score, recall_score, f1_score
from bayes_opt import BayesianOptimization
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
from scikeras.wrappers import KerasClassifier
from sklearn import metrics
import math

pd.set_option("display.max_columns", None)

# %%
score_acc = make_scorer(accuracy_score)

# %%


# %%
def optmize_cnn(dropout_rate, normalization, neurons,optimizer,
                learning_rate, layers1, layers2, neurons1, neurons2, dropout):


    classificador = Sequential()

    classificador.add(Dense(units = neurons, activation='relu', input_dim = 11))
    
    if normalization > 0.5:
        classificador.add(BatchNormalization())

    for i in range(layers1):
        classificador.add(Dense(neurons1, activation='relu'))
        
    if dropout > 0.5:
        classificador.add(Dropout(dropout_rate, seed=123))

    for i in range(layers2):
        classificador.add(Dense(neurons2, activation='relu'))


    classificador.add(Dense(n_classes, activation="softmax"))

    classificador.compile(loss='categorical_crossentropy', 
                    optimizer=optimizer(learning_rate=learning_rate), metrics=['accuracy'])

    return classificador

# %%
def evaluate_network(dropout, normalization, learning_rate, neurons,optimizer, epochs, 
                    batch_size, layer1, layer2, neurons1, neurons2, dropout_rate):
    
    neurons = round(neurons)
    neurons1 = round(neurons1)
    neurons2 = round(neurons2)
    optimizer = round(optimizer)
    epochs = round(epochs)
    batch_size = round(batch_size)
    layer1 = round(layer1)
    layer2 = round(layer2)


    optimizer_array = [Adam, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl, SGD]
    optimizer_val = optimizer_array[optimizer]

    model = optmize_cnn(dropout_rate, normalization, neurons,optimizer_val,
                learning_rate, layer1, layer2, neurons1, neurons2, dropout)

    
    # Train on the bootstrap sample
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    score = []
    
    for train_index, test_index in kf.split(x_train):
        # print("TRAIN:", train_index, "TEST:", test_index)
        xtr, xte = x_train[train_index], x_train[test_index]
        ytr, yte = y_train[train_index], y_train[test_index]

        ytr = pd.get_dummies(ytr)
        yte = pd.get_dummies(yte)

        model.fit(xtr, ytr, epochs=epochs, batch_size=batch_size, verbose=2)

        y_pred = model.predict(xte)
        y_pred = pd.DataFrame(y_pred, columns = [3,4,5,6,7,8])

        y_pred = y_pred.idxmax(axis=1)
        yte = yte.idxmax(axis=1)

        y_pred = (np.asarray(y_pred)).round()
        y_pred = y_pred.astype(int)
        yte = np.array(yte)

        # roc = roc_auc_score(yte, model.predict_proba(xte), multi_class='ovr')
        # acc = accuracy_score(yte, y_pred)
        gm = geometric_mean_score(yte, y_pred, average='macro')
        score.append(gm)

    score = sum(score)/len(score)
    
    return score

# %%
pbounds = {'dropout': (0.0, 0.3),
            'normalization':(0, 1),
            'learning_rate': (0.0, 0.1),
            'neurons': (4, 64),
            'optimizer': (0, 7),
            'epochs' : (50, 300),
            'batch_size': (10, 150),
            'layer1': (1,3),
            'layer2': (1,3),
            'neurons1': (4, 64),
            'neurons2': (4, 64),
            'dropout_rate': (0,0.3)
            }

optimizer = BayesianOptimization(
    f=evaluate_network,
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

optimizer.maximize(init_points=25, n_iter=20,)

# %%
params_nn_ = optimizer.max['params']
learning_rate = params_nn_['learning_rate']
params_nn_['batch_size'] = round(params_nn_['batch_size'])
params_nn_['epochs'] = round(params_nn_['epochs'])
params_nn_['neurons'] = round(params_nn_['neurons'])
optimizerL = [Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl, Adam]
params_nn_['optimizer'] = optimizerL[round(params_nn_['optimizer'])]
params_nn_['layer1'] = round(params_nn_['layer1'])
params_nn_['layer2'] = round(params_nn_['layer2'])
params_nn_['neurons1'] = round(params_nn_['neurons1'])
params_nn_['neurons2'] = round(params_nn_['neurons2'])
params_nn_

# %%
model = optmize_cnn(params_nn_['dropout_rate'], params_nn_['normalization'], params_nn_['neurons'],params_nn_['optimizer'],
                    params_nn_['learning_rate'], params_nn_['layer1'], params_nn_['layer2'], params_nn_['neurons1'],
                    params_nn_['neurons2'], params_nn_['dropout'])


nn = KerasClassifier(model=model, epochs=params_nn_['epochs'], batch_size=params_nn_['batch_size'], verbose=0)

nn.fit(x_train, y_train, verbose=1)

# %%
nn.score(x_test, y_test)

# %%
y

# %%
y_pred = nn.predict(x_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = [3,4,5,6,7,8]


# %%
y_test = y_test.idxmax(axis=1)
y_pred = y_pred.idxmax(axis=1)

# %%
cm = metrics.confusion_matrix(y_test, y_pred)
cm

# %%
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [3,4,5,6,7,8])
cm_display.plot()

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(15,8))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[3,4,5,6,7,8,9], yticklabels=[3,4,5,6,7,8])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)


