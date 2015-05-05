'''
Created on Apr 28, 2015
empty places are labeled as 0, inhabitable are labeled as -np.infs
@author: luchristopher
'''
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import math
from sklearn.preprocessing import OneHotEncoder




def build_matrix(nrows=50,ncols=50,ratio=0.8):
#     base_matrix = np.zeros((nrows,ncols))
#     row_idx,col_idx = np.random.randint(nrows,size=int(0.8*nrows*ncols)),np.random.randint(ncols,size=int(0.8*nrows*ncols))
    base_matrix = np.random.randint(3,size=(nrows,ncols))
    counts = np.bincount(base_matrix.ravel())
    print 'Vacant:Group 1:Group 2: ', counts
    return base_matrix

class MapRoamer():
    
    def __init__(self,value_matrix=np.array([]),nrows=100,ncols=100):    #specifying value matrix
#         if value_matrix.shape == (nrows,ncols):
#             self.V = value_matrix
#         else:
#             raise ValueError()
        pass
        
    #utility functions
    def check_window_boundary(self,base_matrix,row,col,window_size):
        return window_size <= row and base_matrix.shape[0]-window_size > row and window_size <= col and base_matrix.shape[1]-window_size > col
    
    def count_neighbours(self,base_matrix,center_row,center_col,window_size=3,n_groups=3):
        '''
        returns the number of homogeneous and exogenous neighbours in the window
        '''
        neighbour_matrix = base_matrix[center_row-window_size:center_row+window_size+1,center_col-window_size:center_col+window_size+1] #select the window
        counts = np.bincount(neighbour_matrix.ravel(),minlength=n_groups)
        counts[base_matrix[center_row,center_col]] -= 1
#         print counts
#         print neighbour_matrix
        return counts[1:]
        
    def calculate_utility(self,base_matrix,point,window_size,criterion,thresholds):
        '''
        returns the utility of a agent has at the current position
        'linear': the more the homogenous neighbours is, the happier the agent,no need for threshold, instead, threshold are used to pass theta values in an array like shape
                    [Theta_11    Theta_12    Theta_13
                    ...
                    Theta_31    Theta_32    Theta_33]
        'min_max': less happier if > threshold
        'random' : throws out a random preference for each agent under thresholds, 
        'concat' : if more than threshold happiness will remain unchanged
        '''
        utility = -np.inf
        curr_class = base_matrix[point] #find the curr class of people
        neighbour_counts = self.count_neighbours(base_matrix, point[0], point[1], window_size)
#         print neighbour_counts
        if criterion == 'linear':   #threshold are passed as peference theta matrix
#             print 'threshold',thresholds[curr_class-1]
#             print 'counts',neighbour_counts
#             print 'dotproduct: ',thresholds[curr_class-1].dot(neighbour_counts)-np.sum(neighbour_counts)
            return thresholds[curr_class-1].dot(neighbour_counts)-np.sum(neighbour_counts)
        elif criterion == 'min_max':
            return None
        elif criterion == 'concat':
            return None
        elif criterion == 'random':
            curr_pref_vector = np.array(map(lambda x: np.random.uniform(0,x),thresholds[base_matrix[curr_class-1]]))     #calculate pref_theta randomly
            return curr_pref_vector.dot(neighbour_counts)       
            
    def move_proba(self,base_matrix,src_point,dst_point,neighbour_window_size,beta,criterion,threshold_matrix):
        '''
        returns the probability that the agent at i is moving towards j
        '''
        src_util = self.calculate_utility(base_matrix,src_point,neighbour_window_size,criterion,threshold_matrix)
        dst_util = self.calculate_utility(base_matrix,dst_point,neighbour_window_size,criterion,threshold_matrix)
        ext_src = np.exp(beta*src_util)
        ext_dst = np.exp(beta*dst_util)
#         print beta
#         print 'exts: ', ext_src,ext_dst
        return ext_dst/float(ext_src+ext_dst)
    
    ######################################################################    
    #moving functions
    def swap(self,base_matrix,old_row,old_col,new_row,new_col):
        tmp = base_matrix[old_row,old_col]
        base_matrix[old_row,old_col] = base_matrix[new_row,new_col]
        base_matrix[new_row,new_col] = tmp
        
    def move(self,base_matrix,curr_row,curr_col,window_size,beta,criterion,threshold_matrix):
        (max_row,max_col) = base_matrix.shape
        new_row,new_col = np.random.randint(max_row),np.random.randint(max_col)
        while base_matrix[new_row,new_col] != 0 or not self.check_window_boundary(base_matrix, new_row, new_col, window_size):
            new_row,new_col = np.random.randint(max_row),np.random.randint(max_col)
            
        #roll a dice and swap!
#         print 'current dst value:',base_matrix[new_row,new_col]
        p = self.move_proba(base_matrix, (curr_row,curr_col), (new_row,new_col), window_size, beta,criterion,threshold_matrix)
#         print 'migration proba:',p
        if_go = np.random.binomial(1,p)
        if if_go == 1:
            self.swap(base_matrix, curr_row, curr_col, new_row, new_col)

    ########################################################################
    def randomize_threshold(self,window_size):
        '''
        generate randomized satisfication threshold for each individual, ranging from (0,max_in_window)
        '''
        max_num = (2*window_size + 1)**2
        return np.random.randint(max_num/2)
        
            
    def stochastic_roaming(self,base_matrix,max_iter=10000,window_size=1,beta=1,criterion='linear',threshold_matrix=np.array([])):
        '''
        stochastically roam the matrix to relocate agents
        '''
        i = 0
        (max_row,max_col) = base_matrix.shape
        while i < max_iter:
            curr_row,curr_col = np.random.randint(max_row), np.random.randint(max_col)
            if self.check_window_boundary(base_matrix, curr_row, curr_col, window_size) and base_matrix[curr_row,curr_col] != 0:
                self.move(base_matrix, curr_row, curr_col,window_size,beta,criterion,threshold_matrix)  #move function decides if moving will happen for this agent
# #                 print 'moved!!!'
#             else:
#                 print 'unmoved!!!'
                i = i+1
                
    def calculate_satisfication_ratio(self,base_matrix,window_size):
        return base_matrix.apply()
            
def main():
    M = build_matrix()
    print M
#     matshow(M)
#     plt.show()
    roamer = MapRoamer()
    roamer.stochastic_roaming(M, 100000, 1, 1, 'linear', threshold_matrix=np.array([[1,0],[0,0]]))
    print M
    matshow(M,cmap=cm.Greys)
    plt.show()
if __name__ == '__main__':
    main()