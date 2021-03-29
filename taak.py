import numpy as np
import matplotlib.pyplot as plt
import time

#Stap 1) 

data_1 = np.loadtxt('blobs150.csv', delimiter=',')
data_2 = np.loadtxt('blobs1500-1.csv', delimiter=',')
data_3 = np.loadtxt('blobs1500-2.csv', delimiter=',')

plt.scatter(data_1[: ,0] , data_1[: ,1])
plt.scatter(data_2[: ,0] , data_2[: ,1])
plt.scatter(data_3[: ,0] , data_3[: ,1])


def get_distance(x1, x2): # find distance between two vectors using pythagorian theorem
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

def quickSort(arr,non_sort_arr ,low, high):
    if len(arr) == 1:
        return arr
    if low < high:
  
        # pi is partitioning index, arr[p] is now
        # at right place
        pi = partition(arr,non_sort_arr,low, high)
  
        # Separately sort elements before
        # partition and after partition
        quickSort(arr,non_sort_arr ,low, pi-1)
        quickSort(arr,non_sort_arr ,pi+1, high)

def partition(arr,non_sort_arr,low, high):
    i = (low-1)         # index of smaller element
    pivot = arr[high]     # pivot
  
    for j in range(low, high):
  
        # If current element is smaller than or
        # equal to pivot
        if arr[j] <= pivot:
  
            # increment index of smaller element
            i = i+1
            arr[i], arr[j] = arr[j], arr[i]
            non_sort_arr[i] , non_sort_arr[j] =non_sort_arr[j] , non_sort_arr[i]
  
    arr[i+1], arr[high] = arr[high], arr[i+1]
    non_sort_arr[i+1] , non_sort_arr[high] = non_sort_arr[high] ,non_sort_arr[i+1]
    return (i+1)

# Het generiek aanmaken van de matrix-data

## Het aanmaken van de gewogen adjacentie matrix




def new_weighted_adjacency_matrix(data , sigma = 0.31623 ,simularity_function = "exponential_simularity"):
    W = []

    if simularity_function == "exponential_simularity":
        for i in range(len(data)):
            row = []
            for j in range(len(data)):
                row.append(np.exp(-get_distance(data[i],data[j])**2/(2 * sigma**2)))
            W.append(row)


    elif simularity_function == "euclidian_distance":
        for i in range(len(data)):
            row = []
            for j in range(len(data)):
                row.append(np.sqrt(get_distance(data[i],data[j])))
            W.append(row)
    else:
        raise Exception("This simularity function isn't implemented:" + str(simularity_function))

    return W

#print (new_weighted_adjacency_matrix(data_1))
#print (new_weighted_adjacency_matrix(data_1 , simularity_function = "euclidian_distance"))
#print (new_weighted_adjacency_matrix(data_1 , simularity_function = "ezezefeuclidian_distance"))

W = new_weighted_adjacency_matrix(data_1)

#print(W)




## Het aanmaken van de diagonaal matrix

def new_weighted_diagonal_matrix(weighted_adjacency_matrix):
    rowsums = []
    for i in range(len(weighted_adjacency_matrix)):
        rowsums.append(sum(weighted_adjacency_matrix[i]))
    D = np.diagflat(rowsums)
    return D

D = new_weighted_diagonal_matrix(W)
#print(D)


## Het aanmaken van  de gewogen Laplace matrix
def new_weighted_laplace_matrix(weighted_adjacency_matrix,weighted_diagonal_matrix):
    L = weighted_diagonal_matrix - weighted_adjacency_matrix
    return L 

L = new_weighted_laplace_matrix(W,D)
#print(L)


#Het generisch berekenen van de afstand

## Het kmeans algoritme


def display_kmeans_verbose(max_iter, n_points , n_dim , current_iter , begin_time ,n_clusters): 
    print(str(time.ctime()), end =" ")
    print(" finished" , end = " " )
    print(str(current_iter) , end = " ")
    print(" of" , end = " ")
    print(str(max_iter) + " iterations, " , end = " ")
    end_time = time.time()
    delta_time = end_time - begin_time
    projected_time = (max_iter - current_iter) * n_clusters * n_points * n_dim #Time maybe inacurate for small values, 0(t*k*n*d) so O((max-current)*k*n*d)
    print("Projected duration without convergence is : " + str(projected_time) , end = " ")
    print("s")
    return end_time

##For the parameters I looked to the library of scikit learn
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#Hence this is where the defaults where taken from.



def kmeans(data, n_clusters = 3, n_init = 10 , max_iter = 300 ,verbose = 0):
    centroids = data[np.random.choice(range(len(data)), n_clusters, replace=False)] # randomly select k data point
    if verbose > 0:
        print(centroids)
        vieuw_iters = np.floor(np.logspace(0.5, np.log10(max_iter) , verbose , endpoint=True))
        print(vieuw_iters)
        begin_time = time.time()
        n_points = len(data)
        n_dim = len(data[0])
        current_iter_index = 0

    temp = centroids[:]
    converged = False # Flag to terminate process after convergence 
    current_iter = 0

    while (not converged) and (current_iter < max_iter):
        if verbose > 0: #Only show when verbose > 0
            if current_iter == vieuw_iters[current_iter_index]:
                begin_time = display_kmeans_verbose(max_iter, n_points , n_dim , current_iter , begin_time , n_clusters) #displays predicted time returns end time for next verbose
                current_iter_index = current_iter_index + 1 
                

        cluster_list = [[] for i in range(len(centroids))] # cluster for each centeroid
        for x in data:  # Go through each data point
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x)) # get distance to each centeroid
            cluster_list[int(np.argmin(distances_list))].append(x) # add for minimum distance
        cluster_list = list((filter(None, cluster_list))) # remove clusters which are empty         
        prev_centroids = centroids[:] # save centroids to compare later
        centroids = []
        for j in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[j], axis=0)) # calculate the new clusters
        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids)) # get rate of convergence
        converged = (pattern == 0) # check for convergence
        current_iter += 1
        colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
        temp = np.array(temp)
        cluster_list = np.array(cluster_list)
    if verbose > 0:
        print('K-MEANS: ', int(pattern))
        print('number of iterations', current_iter)
    for i in range(len(cluster_list)):
        c = np.array(cluster_list[i])
        plt.scatter(c[:,0],c[:,1],color = colors[i])
        plt.scatter(temp[i,0],temp[i,1],color = colors[i])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return cluster_list #returns the clustered data as a list of lists.

 

#kmeans(data_1, n_clusters = 3, n_init = 10 , max_iter = 100 ,verbose = 0)
#kmeans(data_2, n_clusters = 3, n_init = 10 , max_iter = 100 ,verbose = 0)
#kmeans(data_3, n_clusters = 3, n_init = 10 , max_iter = 100 ,verbose = 0)
#kmeans(data_1, n_clusters = 3, n_init = 10 , max_iter = 100 ,verbose = 10)
#kmeans(data_2, n_clusters = 3, n_init = 10 , max_iter = 100 ,verbose = 10)
#kmeans(data_3, n_clusters = 3, n_init = 10 , max_iter = 100 ,verbose = 10)

def calculate_eigen_vectors_matrix(L , k):
    Eigen_values, Eigen_vectors_right = np.linalg.eig(L)
    n = len(Eigen_values)
    quickSort(Eigen_values,Eigen_vectors_right , 0 , n-1)
    U = []
    for i in range(0,k):
        U.append(Eigen_vectors_right[i])
    U = np.array(U)
    U = np.transpose(U) 
    return U

def unormalized_spectral_clustering(data ,simularity_function ,n_clusters):
    W = new_weighted_adjacency_matrix(data , simularity_function=simularity_function)
    D = new_weighted_diagonal_matrix(W)
    L = new_weighted_laplace_matrix(W,D)
    U = calculate_eigen_vectors_matrix(L , n_clusters)
    kmeans(U, n_clusters = 3, n_init = 10 , max_iter = 300 ,verbose = 0)
    return 0
kmeans(data_1, n_clusters = 3, n_init = 10 , max_iter = 300 ,verbose = 0)  
unormalized_spectral_clustering(data_1 ,"exponential_simularity" ,3)






# Function to do Quick sort
#https://www.geeksforgeeks.org/python-program-for-quicksort/ 
