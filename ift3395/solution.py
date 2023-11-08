import numpy as np
import matplotlib.pyplot as plt
winequality = np.genfromtxt('winequality.txt')

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, wineQuality):
        return np.mean(wineQuality[:,:-1], axis=0)
    
    def covariance_matrix(self, wineQuality):
        return np.cov(wineQuality[:,:-1], rowvar= False)

    def feature_means_class_5(self, wineQuality):
        condition_5 = wineQuality[:,-1]==5
        return self.feature_means(wineQuality[condition_5])

    def covariance_matrix_class_5(self, wineQuality):
        condition_5 = wineQuality[:,-1]==5
        return self.covariance_matrix(wineQuality[condition_5])

#This method is not generalized and works on numeric classes only!
# class HardParzen:
#     def __init__(self, h):
#         self.h = h

#     def train(self, train_inputs, train_labels):
#         self.train_inputs = train_inputs
#         self.train_labels = train_labels
#         self.label_list = np.unique(train_labels)
#         self.nb_classes = len(self.label_list)

#     def compute_predictions(self, test_data):
#         # Initialization of the count_matrix and the predicted classes array
#         num_test = test_data.shape[0]
#         counts_matrix = np.zeros((num_test, self.nb_classes))
#         classes_pred = np.zeros(num_test)
#         offset_index = int(np.min(self.label_list))
#         # for each data point find the distances to each training set point using eucidian distance
#         for (i, ex) in enumerate(test_data):
#             distances = np.linalg.norm(ex - self.train_inputs,axis=1)
#             #distances = np.sqrt(np.sum((ex-self.train_inputs)**2,axis=1))
#         # Go through the training set to find the neighbors of the current point (ex) within the radius h
#             neighbors = np.where(distances<=self.h)[0]   
#         #If I can't find a neighbor then I will predict a label randomly    
#             if(len(neighbors)==0):
#                 index = int(draw_rand_label(ex, self.label_list))
#                 counts_matrix[i,index-offset_index]+=1
#         # Calculate the number of neighbors belonging to each class and write them in counts_matrix
#             else:
#                 for index in neighbors:
#                     counts_matrix[i,int(self.train_labels[index])-offset_index]+=1    
#         # From the counts_matrix, we will get the classes_pred[i] 
#             classes_pred[i] = np.argmax(counts_matrix[i,:]) + offset_index  
#         return classes_pred
    
#This method works with indexes so the classes can be anything (numerci or otherwise)
class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.nb_classes = len(self.label_list)

    def compute_predictions(self, test_data):
        # Initialization of the count_matrix and the predicted classes array
        num_test = test_data.shape[0]
        counts_matrix = np.zeros((num_test, self.nb_classes))
        classes_pred = np.zeros(num_test)
    
        # for each data point find the distances to each training set point using eucidian distance
        for (i, ex) in enumerate(test_data):
            distances = np.linalg.norm(ex - self.train_inputs,axis=1)
            #distances = np.sqrt(np.sum((ex-self.train_inputs)**2,axis=1))
        # Go through the training set to find the neighbors of the current point (ex) within the radius h
            neighbors = np.where(distances<=self.h)[0]   
        #If I can't find a neighbor then I will predict a label randomly    
            if(len(neighbors)==0):
                random_label = draw_rand_label(ex, self.label_list)
                index = np.where(self.label_list==random_label)[0]
                counts_matrix[i,index]+=1
        # Calculate the number of neighbors belonging to each class and write them in counts_matrix
            else:
                for index in neighbors:
                    counts_matrix[i,np.where(self.label_list==self.train_labels[index])[0][0]]+=1    
        # From the counts_matrix, we will get the classes_pred[i] 
            classes_pred[i] = self.label_list[np.argmax(counts_matrix[i,:])]
            
        return classes_pred

class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma
        
    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.nb_classes = len(self.label_list)

    def compute_predictions(self, test_data):
        # Initialization of the count_matrix and the predicted classes array
        num_test = test_data.shape[0]
        classes_pred = np.zeros(num_test)
        #Find the index at which the data labels start
        offset_index = int(np.min(self.label_list))
        #Calculate d for the Gaussian kernel function 
        d = self.train_inputs.shape[1]
        #For each point in test_data, we compute the Gaussian kernel between it and train_inputs using Euclidean distance. 
        for (i, ex) in enumerate(test_data):
            voting_points = np.zeros(len(self.label_list))
            distances = np.linalg.norm(ex - self.train_inputs,axis=1)
            #distances = np.sqrt(np.sum((ex-self.train_inputs)**2,axis=1))
            #Gaussian_kernels =  (1/((2*np.pi)**(d/2)*(self.sigma**d)))*np.exp(-(distances**2/(2*(self.sigma**2))))
            Gaussian_kernels = np.divide(1,np.multiply(2*np.pi,np.power(self.sigma,d/2)))*np.exp(-np.divide(np.square(distances),2*np.square(self.sigma)))
        #Filling the voting_weights array with the Gausian_kernels voting points
            for j in range(len(Gaussian_kernels)):
                voting_points[int(self.train_labels[j])-offset_index] += Gaussian_kernels[j]
        # From the voting_points array, we will define classes_pred[i]
            classes_pred[i] = np.argmax(voting_points) + offset_index
        return classes_pred


def split_dataset(wineQuality): 
    train = wineQuality[np.arange(wineQuality.shape[0])%5 <3]
    validation = wineQuality[np.arange(wineQuality.shape[0])%5==3]
    test =wineQuality[np.arange(wineQuality.shape[0])%5==4]
    return (train,validation,test)


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        #Create an instance of the class HardParzen with the radius h
        HardParzen_func = HardParzen(h)
        #Train the HardParzen model
        HardParzen_func.train(self.x_train,self.y_train)
        #Get the predictions array
        pred_classes = HardParzen_func.compute_predictions(self.x_val)
        numberOfErrors = 0
        numberOfTests = len(self.y_val)
        #Compare and count the number of errors and return the percentage of errors over the size of the data
        for i in range(numberOfTests):
            if(pred_classes[i]!=self.y_val[i]):
                numberOfErrors+=1
        return float(numberOfErrors)/numberOfTests


    def soft_parzen(self, sigma):
        #Create an instance of the class SoftRBFParzen with the parametre sigma
        SoftRBFParzen_func = SoftRBFParzen(sigma)
        #Train the SoftRBFParzen model
        SoftRBFParzen_func.train(self.x_train,self.y_train)
        #Get the predictions array
        pred_classes = SoftRBFParzen_func.compute_predictions(self.x_val)
        numberOfErrors = 0
        numberOfTests = len(self.y_val)
        
        #Compare and count the number of errors and return the percentage of errors over the size of the data
        for i in range(numberOfTests):
            if(pred_classes[i]!=self.y_val[i]):
                numberOfErrors+=1     
        return float(numberOfErrors)/numberOfTests
        


def get_test_errors(wineQuality):
    train,validation,test = split_dataset(wineQuality)
    ErrorTester = ErrorRate(train[:,:-1],train[:,-1],validation[:,:-1],validation[:,-1])
    smallest_error_h = float('inf')
    smallest_error_sigma = float('inf')
    best_h = 0
    best_sigma= 0
    for h in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]:
        if(ErrorTester.hard_parzen(h)<smallest_error_h):
            smallest_error_h = ErrorTester.hard_parzen(h)   
            best_h = h 
    for sigma in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]:
        if(ErrorTester.soft_parzen(sigma)<smallest_error_sigma):
            smallest_error_sigma = ErrorTester.soft_parzen(sigma)   
            best_sigma = sigma
    New_Tester = ErrorRate(train[:,:-1],train[:,-1],test[:,:-1],test[:,-1])        
        
    return [New_Tester.hard_parzen(best_h),New_Tester.soft_parzen(best_sigma)]

def get_val_errors(wineQuality):
    train, validation, test = split_dataset(wineQuality)
    ErrorTester = ErrorRate(train[:, :-1], train[:, -1], validation[:, :-1], validation[:, -1])

    # Lists to store error rates for Hard Parzen and Soft Parzen
    hard_parzen_errors = []
    soft_parzen_errors = []

    for h in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]:
        hard_error = ErrorTester.hard_parzen(h)
        hard_parzen_errors.append(hard_error)

    for sigma in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]:
        soft_error = ErrorTester.soft_parzen(sigma)
        soft_parzen_errors.append(soft_error)

    return [hard_parzen_errors, soft_parzen_errors]




def random_projections(X, A):
    #Check if we are able to do the multiplication (nbr de colonnes de X doit etre = nbr de lignes de A)
    if X.shape[1] != A.shape[0]:
        return np.zeros((X.shape[1], A.shape[1]))
    else:
        return np.divide(1,np.sqrt(2))*X.dot(A)


'''train,validation,test = split_dataset(winequality)
HardParzen_func = HardParzen(5)
HardParzen_func.train(train[:,:-1],train[:,-1])
print(HardParzen_func.compute_predictions(test[:,:-1]))'''


'''SoftRBFParzen_func = SoftRBFParzen(0.5)
SoftRBFParzen_func.train(train[:,:-1],train[:,-1])
print(SoftRBFParzen_func.compute_predictions(test[:,:-1]))'''

#ErrorTester = ErrorRate(train[:,:-1],train[:,-1],validation[:,:-1],validation[:,-1])
#print(ErrorTester.hard_parzen(1))
#print(ErrorTester.soft_parzen(0.1))

# Define the values of h and σ to test

# Define the values of h and σ to test
h_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
sigma_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]

# Calculate error rates for Hard Parzen for each value of h
hard_parzen_errors = get_val_errors(winequality)[0]

# Calculate error rates for Soft Parzen for each value of σ
soft_parzen_errors = get_val_errors(winequality)[1]

# Create a single graph with two curves
plt.figure(figsize=(10, 6))
plt.plot(h_values, hard_parzen_errors, marker='o', label='Hard Parzen')
plt.plot(sigma_values, soft_parzen_errors, marker='o', label='Soft Parzen')

# Set labels and title
plt.xlabel('h and σ')
plt.ylabel('Error Rate')
plt.title('Error Rates for Hard Parzen and Soft Parzen')
plt.yscale('log')  # Use a logarithmic scale for better visualization of small values

plt.xscale('log')

# Show legend
plt.legend()

# Display the graph
plt.grid(True)
plt.show()




