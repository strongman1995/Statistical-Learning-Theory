
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import pickle as pkl
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

def load_data(filename):
    with open(filename, 'r') as f:
        data = f.readlines()  #txt中所有字符串读入data
        y = np.zeros(len(data))
        X = np.zeros([len(data),123])
        for i,line in enumerate(data):
            line_list = line.strip().split(' ')        #将单个数据分隔开存好
            y[i] = line_list[0]
            for fea in line_list[1:]: #第一列是label
                feature_id,feature = fea.split(':')
                X[i][int(feature_id)-1] = feature
    return X,y

X_train, y_train = load_data('a9a')
X_test, y_test = load_data('a9a.t')
print len(y_train),X_train.shape
print len(y_test),X_test.shape


# In[5]:

y_train[y_train < 0] = 0
y_test[y_test < 0] = 0


# In[7]:

# 从train中分出15%给dev集
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
X_train.shape, X_dev.shape, y_train.shape, y_dev.shape


# In[8]:

def gradient_L(X,y,w):
    mu = sigmoid(X.dot(w))
    grad_L = (y-mu).dot(X)
    return grad_L


# In[9]:

def calc_H(X,w):
    u = sigmoid(X.dot(w))
    m, n = X.shape
    R = np.eye(m) * u * (1 - u)
    H = -X.T.dot(R).dot(X)
    return H


# In[10]:

def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


# In[11]:

def get_proba(X, w):
    prob_y = sigmoid(X.dot(w))
#     prob_y[prob_y > 0.5] = 1
#     prob_y[prob_y < 0.5] = 0
    return prob_y


# In[13]:

def get_accuracy(y, prob_y, threshold = 0.5):
    temp = np.ones(len(y))
    prob_y[prob_y > threshold] = 1
    prob_y[prob_y < threshold] = 0
    correct_cnt = np.sum(temp[y==prob_y])
    return 1.0*correct_cnt/len(y)


# In[14]:

def L(X, y, w):
    return y.dot(X.dot(w)) - np.sum(np.log(1+np.exp(X.dot(w))))


# In[15]:

def l2_L(X, y, w, lbd):
    return -0.5*lbd*np.linalg.norm(w)+L(X, y, w)


# In[80]:

def l2_norm_IRLS(X, y, lbd, max_iter=50, loss=1e-3):
    m,n = X.shape
    wt = np.array([1e-13]*n)
    accuracy_list = list()
    l2_norm_wt_list = list()
    loss_list = list()
    delta = 1
    print "="*10, " lambda ", lbd, "="*10
    f=open('log.txt','w')
    f.write("="*10+str(lbd)+"="*10+"\n")
    for i in range(max_iter):
        prob_y = sigmoid(X.dot(wt))
        prob_y[prob_y > 0.5] = 1
        prob_y[prob_y < 0.5] = 0
#         auc_score = roc_auc_score(y, prob_y)
        accuracy = accuracy_score(y, prob_y)
            
        accuracy_list.append(accuracy)
        l2_norm_wt_list.append(np.linalg.norm(wt))
                
        u = sigmoid(X.dot(wt))
        R = np.eye(m) * u * (1 - u)
        H = -X.T.dot(R).dot(X)-lbd*np.eye(n)
        grad_L = X.T.dot(y-u)-lbd*wt
        
        wt_ = wt-np.linalg.pinv(H).dot(grad_L)
        
        delta = np.linalg.norm(l2_L(X, y, wt_, lbd)-l2_L(X, y, wt, lbd))
        loss_list.append(delta)
        print "iter_num:", i+1, "L:", l2_L(X, y, wt_, lbd),"accuracy:", accuracy, "loss:", delta
        if delta < loss:
            break
        f.write("iter_num:"+ str(i+1)+"L:"+str(l2_L(X, y, wt_, lbd))+"accuracy:"+str(accuracy)+"loss:"+str(delta)+"\n")
        wt = wt_
    f.close()
    
#     plot_prec_recall_curve(y_test=y, y_prob=prob_y)
    return wt, accuracy_list, l2_norm_wt_list,loss_list


# In[69]:

def test(X,y,w):
    prob_y = sigmoid(X.dot(w))
    prob_y[prob_y > 0.5] = 1
    prob_y[prob_y < 0.5] = 0
    return accuracy_score(y, prob_y)


# In[93]:

# lambda=0 没有加正则项
# lambda>0 加了正则项
plot_train = []
lbd_list = [0,0.01,0.1,1,10,100]
# for lbd in [0,0.01,0.1,1,10,100]:
for lbd in lbd_list:
    result = l2_norm_IRLS(X_train, y_train, lbd, max_iter=50, loss=1e-4)
    plot_train.append(result)
print "Complete train!"   


# In[95]:

accuracy_score_list = list()
for wt, accuracy_list, l2_norm_wt_list, loss_list in plot_train:
    accuracy_score_dev = test(X_dev, y_dev, wt)
    accuracy_score_list.append(accuracy_score_dev)

max_index = accuracy_score_list.index(max(accuracy_score_list))
best_wt = plot_train[max_index][0]
accuracy_score_test = test(X_test, y_test, best_wt)


# In[103]:

accuracy_score_test


# In[97]:

for i, (wt, accuracy_list, l2_norm_wt_list, loss_list) in enumerate(plot_train):
    plt.plot(accuracy_list,label="$\lambda = $"+str(lbd_list[i]))
    plt.legend(loc=0,prop={'size': 6})
    plt.xlabel('iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy curve in training set')
plt.savefig("Accuracy_train.jpg")  


# In[101]:

for i, (wt, accuracy_list, l2_norm_wt_list, loss_list) in enumerate(plot_train):
    plt.plot(l2_norm_wt_list,label="$\lambda = $"+str(lbd_list[i]))
    plt.legend(loc=0,prop={'size': 6})
    plt.xlabel('iteration')
    plt.ylabel('l2-norm of $\|w\|_2$')
    plt.title('l2-norm of $\|w\|_2$ curve in training set')
plt.savefig("l2_norm_w_train.jpg")  


# In[100]:

for i, (wt, accuracy_list, l2_norm_wt_list, loss_list) in enumerate(plot_train):
    plt.plot(loss_list,label="$\lambda = $"+str(lbd_list[i]))
    plt.legend(loc=0,prop={'size': 6})
    plt.xlabel('iteration')
    plt.ylabel('loss$')
    plt.title('loss curve in training set')
plt.savefig("loss_curve_train.jpg")  

