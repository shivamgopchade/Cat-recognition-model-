import numpy as np
import lr_utils

train_set_x_org,train_set_y,test_set_x_org,test_set_y,classes= lr_utils.load_dataset()

#plt.imshow(train_set_x_org[25])
#plt.show()

m_train=train_set_x_org.shape[0]
m_test=train_set_x_org.shape[0]
num_px=train_set_x_org[0].shape[0]

train_set_x_flatten=train_set_x_org.reshape(train_set_x_org.shape[0],-1).T
test_set_x_flatten=test_set_x_org.reshape(test_set_x_org.shape[0],-1).T

train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

b=0
w=np.zeros((train_set_x.shape[0],1))



def sigmoid(z):
    return 1/(1+np.exp(-z))

def propogate(X,Y,w,b):
    z=np.dot(w.T,X)+b
    A=sigmoid(z)
    m=X.shape[1]


    J=(-1/m)*(np.dot(Y[0],(np.log(A[0]).T))+np.dot(1-Y[0],(np.log(1-A[0])).T))

    dz=A-Y

    grad_w=(1/m)*(np.dot(X,(dz.T)))
    grad_b=np.sum(dz)/m

    grads={'dw':grad_w,'db':grad_b}
    return J,grads

def optimize(X,Y,w,b,lr,iter,prt=False):

    costs=[]
    for i in range(iter):
        J,grads=propogate(X,Y,w,b)
        dw=grads['dw']
        db=grads['db']

        w=w-lr*dw
        b=b-lr*db

        if i%100==0:
            costs.append(J)

    if prt:
        print(costs)
    return {'w':w,'b':b},{'dw':dw,'db':db},costs

def predict(w,b,X):
    z=np.dot(w.T,X)+b
    A=sigmoid(z)
    #Y_predict=np.zeros((1,X.shape[1]))
    Y_predict=A>=0.5
    return Y_predict


def model(X_train, Y_train, X_test, Y_test, num_iter, lr, prt=False):
    b = 0
    w = np.zeros((train_set_x.shape[0], 1))
    param, grads, costs = optimize(X_train, Y_train, w, b, lr, num_iter, prt)
    w=param['w']
    b=param['b']

    Y_predict_train=predict(w,b,X_train)
    Y_predict_test=predict(w,b,X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_predict_test,
         "Y_prediction_train": Y_predict_train,
         "w": w,
         "b": b,
         "learning_rate": lr,
         "num_iterations": num_iter}

    return d
#w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iter = 2000, lr = 0.005, prt = False) #model example
