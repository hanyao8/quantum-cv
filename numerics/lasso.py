import time
from scipy import sparse
from scipy import linalg

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets

from qubovert.sim import anneal_qubo
from qubovert import boolean_var

########################################################################
gamma=0.02

########################################################################


X_all, y_all = datasets.load_diabetes(return_X_y=True)
print(X_all.shape)

X_train = X_all[:150]
y_train = y_all[:150]

X = X_all[:150]
y = y_all[:150]

N = X.shape[0]
D = X.shape[1]

print(X)
print(y)

X_val = X_all[150:]
y_val = y_all[150:]

X_mean = np.mean(X,axis=0)
X_std = np.std(X,axis=0)
y_mean = np.mean(y)
y_std = np.std(y)

X = (X-X_mean)/X_std
y = (y-y_mean)/y_std

########################################################################

#alphas = np.logspace(-100, 2, 30)
alphas = np.logspace(-3, 0, 15)

scores_train = []
scores_val = []
plot_y = []

for i in range(len(alphas)):
    print(i)
    print("alpha= %f"%alphas[i])
    lasso = Lasso(alpha=alphas[i],random_state=0, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_train_pred = np.matmul(X_train,lasso.coef_)+lasso.intercept_
    y_val_pred = np.matmul(X_val,lasso.coef_)+lasso.intercept_
    r2_train = lasso.score(X_train,y_train)
    r2_val = lasso.score(X_val,y_val)

    SSres_train = np.dot(y_train_pred-y_train,y_train_pred-y_train)
    SSres_val = np.dot(y_val_pred-y_val,y_val_pred-y_val)

    SStot_train = np.dot(y_mean-y_train,y_mean-y_train)
    SStot_val = np.dot(np.mean(y_val)-y_val,np.mean(y_val)-y_val)

    r2_train_calc = 1-SSres_train/SStot_train
    r2_val_calc = 1-SSres_val/SStot_val

    one_norm = np.matmul(np.abs(lasso.coef_),np.ones(D))
    zero_norm = np.matmul(np.where(np.abs(lasso.coef_)>0,1,0),np.ones(D))

    #scores.append(r2_val)
    scores_train.append(r2_train)
    scores_val.append(r2_val)
    #scores_train.append(sqloss_train)
    #scores_val.append(sqloss_val)
    plot_y.append(zero_norm)
    
    print("w= %s"%(str(lasso.coef_)))
    print("r2_val= %f"%r2_val)
    print("r2_val_calc= %f"%r2_val_calc)
    print("sqloss_val= %f"%SSres_val)
    print("one_norm= %f"%one_norm)
    print("zero_norm= %f"%zero_norm)
    print("\n")

fig1 = plt.figure(figsize=(6,4))
ax1 = fig1.add_subplot(111)

#plt.rcParams['figure.figsize'] = [8, 5]
#ax1.plot(np.log10(alphas),1-np.array(scores_train),label="train")
#ax1.plot(np.log10(alphas),1-np.array(scores_val),label="val 1-fold fixed")
ax1.plot(np.log10(alphas),plot_y)
ax1.set_title("Diabetes dataset (D=10), sklearn")
ax1.set_xlabel("log10(alpha)")
#ax1.set_ylabel("1-r2 score")
ax1.set_ylabel("|w|_0 zero norm")
ax1.grid()
ax1.legend()
fig1.savefig('lasso_f1.png')
#plt.show()

print("conventional lasso finished")
lasso = None
print("*"*128)
print("\n")

########################################################################


def update_m(X,y,w,alpha):
    m = {i: boolean_var('m(%d)' % i) for i in range(D)}

    A = np.linalg.multi_dot([np.diag(w),X.T,X,np.diag(w)])
    b = -2*np.linalg.multi_dot([np.diag(w),X.T,y])
    b = b + alpha*w*np.sign(w)

    model = 0
    for i in range(D):
        for j in range(D):
            model += m[i]*(A[i][j]+1e-9)*m[j]
        model += (b[i]+1e-9)*m[i]
        
    time_start = time.time()
    res = anneal_qubo(model, num_anneals=10)
    #print("Anneal time taken %f"%(time.time()-time_start))
    model_solution = res.best.state
    #print("Model value:", res.best.value)
    #print("Model value + yT.y:", res.best.value+np.dot(y,y))
    
    m = np.array(list(model_solution.values()))
    return(m)


def predict(mw):
    X_all, y_all = datasets.load_diabetes(return_X_y=True)
    print(X_all.shape)

    X_train = X_all[:150]
    y_train = y_all[:150]

    X = X_all[:150]
    y = y_all[:150]

    N = X.shape[0]
    D = X.shape[1]

    print(X)
    print(y)

    X_val = X_all[150:]
    y_val = y_all[150:]

    X_mean = np.mean(X,axis=0)
    X_std = np.std(X,axis=0)
    y_mean = np.mean(y)
    y_std = np.std(y)

    X = (X-X_mean)/X_std
    y = (y-y_mean)/y_std

    y_train_pred = np.matmul(X,mw)*y_std+y_mean
    y_val_pred = np.matmul((X_val-X_mean)/X_std,mw)*y_std+y_mean
    SSres_train = np.dot(y_train_pred-y_train,y_train_pred-y_train)
    SSres_val = np.dot(y_val_pred-y_val,y_val_pred-y_val)

    SStot_train = np.dot(y_mean-y_train,y_mean-y_train)
    SStot_val = np.dot(np.mean(y_val)-y_val,np.mean(y_val)-y_val)

    r2_train = 1-SSres_train/SStot_train
    r2_val = 1-SSres_val/SStot_val
    
    one_norm = np.matmul(np.abs(mw),np.ones(D))
    zero_norm = np.matmul(np.where(np.abs(mw)>0,1,0),np.ones(D))
    
    scores_train.append(r2_train)
    scores_val.append(r2_val)
    plot_y.append(zero_norm)
    
    print("m*w= %s"%(str(mw)))
    print("r2_val= %f"%r2_val)
    print("sqloss_val= %f"%SSres_val)
    print("one_norm= %f"%one_norm)
    print("zero_norm= %f"%zero_norm)
    print("\n")


def custom_algo(X,y,alpha,gamma=0.02,max_iters=100):
    #max_time
    #alpha = 20.0
    #gamma=0.5

    #N = X.shape[0]
    D = X.shape[1]

    m_prev = np.ones(D)
    m = np.ones(D)
    #print(m)
    w_prev = np.zeros(D)
    w = np.linalg.multi_dot([np.diag(m),X.T,X,np.diag(m)])
    w = np.linalg.multi_dot([np.linalg.inv(w),np.diag(m),X.T,y])
    #print(w)

    t = 1
    step_change = np.linalg.norm(m*w-m_prev*w_prev)
    while step_change > 0.001:
    #alternatively: use time condition. "Must not spend more than x seconds"
        #print(t)
        print("iter no.:")
        print(t)
        m_prev = m.copy()
        m = update_m(X,y,w,alpha)
        #print(m_prev)
        #print(m)
        
        w_prev = w.copy()
        w = np.linalg.multi_dot([np.diag(m),X.T,X,np.diag(m)])
        w = np.linalg.multi_dot([np.linalg.inv(w+np.eye(D)*1e-9),np.diag(m),X.T,y])
        w = (1-gamma)*w_prev + gamma*w
        # w itself should not be sparse.
        # It should be the optimal weights given some sparsity defined by m.
        #w = (1-gamma*m)*w_prev + gamma*m*w
        #w = (1-gamma)*w_prev + gamma*m*w + gamma*(1-gamma)*(1-m)*w
        #print(w_prev)
        #print(w)
        #predict(m*w)
        
        step_change = np.linalg.norm(m*w-m_prev*w_prev)
        t += 1
        #print(step_change)
        #print('\n')
        if t>max_iters:
            break

    print('iters taken: %d'%t)

    print("end of custom algo run:")
    print("m:%s"%str(m))
    print("w:%s"%str(w))
    return(m*w) 
    #should return m*w because in the overall formulation 
    #it is m*w that is optimized wrt the objective.
    #return(w)


#alphas = np.logspace(-1, 2, 30)
alphas = np.array([15.0])

scores_train = []
scores_val = []
plot_y = []

for i in range(len(alphas)):
    print(i)
    print("alpha= %f"%alphas[i])

    mw = custom_algo(X,y,alphas[i],gamma)
    y_train_pred = np.matmul(X,mw)*y_std+y_mean
    y_val_pred = np.matmul((X_val-X_mean)/X_std,mw)*y_std+y_mean
    SSres_train = np.dot(y_train_pred-y_train,y_train_pred-y_train)
    SSres_val = np.dot(y_val_pred-y_val,y_val_pred-y_val)

    SStot_train = np.dot(y_mean-y_train,y_mean-y_train)
    SStot_val = np.dot(np.mean(y_val)-y_val,np.mean(y_val)-y_val)

    r2_train = 1-SSres_train/SStot_train
    r2_val = 1-SSres_val/SStot_val
    
    one_norm = np.matmul(np.abs(mw),np.ones(D))
    zero_norm = np.matmul(np.where(np.abs(mw)>0,1,0),np.ones(D))
    
    scores_train.append(r2_train)
    scores_val.append(r2_val)
    plot_y.append(zero_norm)
    
    print("m*w= %s"%(str(mw)))
    print("r2_val= %f"%r2_val)
    print("sqloss_val= %f"%SSres_val)
    print("one_norm= %f"%one_norm)
    print("zero_norm= %f"%zero_norm)
    print("\n")

fig2 = plt.figure(figsize=(6,4))
ax2 = fig2.add_subplot(111)

#plt.rcParams['figure.figsize'] = [8, 5]
#ax2.plot(np.log10(alphas),1-np.array(scores_train),label="train")
#ax2.plot(np.log10(alphas),1-np.array(scores_val),label="val 1-fold fixed")
ax2.plot(np.log10(alphas),plot_y)
ax2.set_title("Diabetes dataset (D=10), QUBO sparse reg")
ax2.set_xlabel("log10(alpha)")
#ax2.set_ylabel("1-r2 score")
ax2.set_ylabel("|w|_0 zero norm")
ax2.grid()
ax2.legend()
fig2.savefig('lasso_f2.png')
