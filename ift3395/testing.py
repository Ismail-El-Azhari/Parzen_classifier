import numpy as np
winequality = np.genfromtxt('winequality.txt')
'''tab = np.array([[3,5],[6,-9],[5,44],[2,8]])
condition = tab[:,-1]>=5
print(tab[condition])'''

'''list1=np.array(['a','b'])
list2=np.array([3,4])
l1 = np.vstack((list1,list2))
l2 = np.hstack((list1,list2))
print(l1)
print(l2)'''

tab= np.array([[5,4,3],[1,2,3]])
'''print(tab.shape)
print(tab.shape[0])'''
'''for (i ,ex) in enumerate(tab):
    print(ex)'''
train_inputs = np.array([[5,2],[3,3],[6,10]])
test_data= np.array([[10,10],[20,20]])
indexes =[]

###################
tab1 = np.array([4,9,3,6,10])
tab2=np.where(tab1<5)[0]
#print(tab2)

'''for (i,ex) in enumerate(test_data):
    differences = np.sum(np.abs(ex - train_inputs), axis=1)
    #print(i)
    #indexes.append(np.where(differences<10)[0])
    #indexes = np.where(differences>5)[0]
    #print(indexes)
    #Gkernel = (differences**2) -100
    #print(Gkernel)
for i in indexes:
    print(i)'''
'''def split_dataset(wineQuality): 
    train = wineQuality[np.arange(wineQuality.shape[0])%5 <3]
    validation = wineQuality[np.arange(wineQuality.shape[0])%5==3]
    test =wineQuality[np.arange(wineQuality.shape[0])%5==4]
    return (train,validation,test)
#print(split_dataset(winequality)[0,:])
tab = np.array([0.0,0.0,0.0])
if np.all(tab == np.zeros(len(tab))):
    print('yes')'''

tab = np.array([[1,1],[2,2],[5,6]])
A = np.array([[0,0],[1,10],[2,20]])
point= [2,2]
print(np.linalg.norm(point- tab,axis=1))

def random_projections(X, A):
    #Si le nbr de ligne de X (col de Xt)!= nbr de ligne de A, on retourne une matric vide de la bonne taille
    if X.shape[0] != A.shape[0]:
        return np.zeros((X.shape[0], A.shape[1]))
    elif A.shape != [11,2] or X.shape[1] != 11:
        return np.zeros((X.shape[1], A.shape[1]))
    else:
        return np.divide(1,np.sqrt(2))*X.dot(A)

A = np.array([[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2]])
#print(np.divide(1,np.sqrt(2))*4)
#print(random_projections(winequality[:,:-1],A))
#rint(random_projections(tab,A))

