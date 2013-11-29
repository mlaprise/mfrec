'''
    Cython module
    
    Training function for the Kernel Matrix Factorization
    with a logistic kernel using a Stochastic Gradient Descent
    
    Created on July 24th, 2011
    
    @author: Martin Laprise
    
'''

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, exp
import cython
cimport numpy as np


ctypedef np.float64_t DTYPE_t


cdef double clamping(double value, double min = 1.0, double max = 5.0):
    '''
    Clamp the value between min and max
    '''
    if value > 5.0:
        value = 5.0 
    if value < 1.0:
        value = 1.0
        
    return value


cdef double estimator(int item_index,
                      int user_index,
                      int f,
                      double u,
                      double v,
                      int dim,
                      double f_init,
                      double cache = 0.0,
                      int trailing = 0,
                      double overall_avg = 1.0,
                      double item_bias = 0.0,
                      double user_bias = 0.0):
    '''
    Estimate the rating of known user-item pair (used in the GD algo)
    Do not loop over all the feature, uses the cache value for the training.

    Uses the sum of all bias (overall, users and items as a baseline). Those bias
    are only evaluated at the beginning only => Not for bias learning
    '''
    
    cdef double sum
    
    if cache > 0:
        sum = cache
    else:
        sum = overall_avg + item_bias + user_bias

    sum += u * v

    sum = clamping(sum)
        
    if trailing == 1:
        sum += (dim - f - 1) * f_init * f_init
        sum = clamping(sum)
        
    return sum


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)  
cdef double full_estimator(int item_index,
                        int user_index,
                        int f,
                        np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                        np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                        int dim,
                        double overall_avg = 0.0,
                        double item_bias = 0.0,
                        double user_bias = 0.0):
    '''
    Estimate the rating of known user-item pair (used in the GD algo)
    This version loop over all the feature and recompute the bias at each time (do not uses the cache)
    '''
    
    cdef DTYPE_t sum
    cdef int feature

    sum = overall_avg + item_bias + user_bias 
    
    for feature in xrange(dim):
        sum += u[feature, item_index] * v[feature, user_index] 
        
    return sum


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)      
def train_logistic_kernel(int nbr_epochs,
                          int dim,
                          double f_init,
                          double learning_rate,
                          double learning_rate_users,
                          double learning_rate_items,
                          double K_users,
                          double K_items,
                          double K_bias,
                          double overall_avg,
                          np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                          np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                          np.ndarray[np.int32_t, ndim=2, mode="c"] ratings_index,
                          np.ndarray[np.float64_t, ndim=1, mode="c"] ratings,
                          np.ndarray[np.float64_t, ndim=1, mode="c"] items_bias,
                          np.ndarray[np.float64_t, ndim=1, mode="c"] users_bias,
                          int update_users = 1,
                          int update_items = 1,
                          int verbose = 0):
    '''
    Main Loop for the Stochastic Gradient Descent training process for the Kernel Matrix Factorization
    method with a logistic kernel
    '''
    
    cdef double rmse = 2.0
    cdef double rmse_last  = 0.0
    cdef double error = 0.0
    cdef double squared_error = 0.0
    cdef double cf = 0.0
    cdef double mf = 0.0
    cdef double rating = 0.0
    cdef double p = 0.0
    cdef double dot_prod, sig_dot = 0.0
    cdef double rating_range_size = 4.0
    cdef double min_rating = 1.0
    cdef double grad = 0.0

    cdef int epoch = 0
    cdef int user_index = 0
    cdef int item_index = 0
    cdef int f = 0
    cdef long i = 0
    cdef long nbr_ratings = 0
    
    nbr_ratings = ratings.shape[0]
    
    for epoch in xrange(nbr_epochs):
        squared_error = 0.0
        for i in xrange(nbr_ratings): 
            # Retrieves the rating
            user_index = ratings_index[i,0]
            item_index = ratings_index[i,1]
            rating = ratings[i]

            # Performs the estimation
            dot_prod = full_estimator(item_index, user_index, f, u, v, dim, 0.0, items_bias[item_index], users_bias[user_index])
            sig_dot = 1.0 / (1.0 + exp(-dot_prod))          
            p = min_rating + sig_dot * rating_range_size

            # Compute the error on the prediction and set the gradient accordingly 
            error = (rating - p);
            squared_error += error * error
            grad = error * sig_dot * (1.0 - sig_dot) * rating_range_size

            # Update the users and items bias
            if update_users:
                users_bias[user_index] += learning_rate * (grad - K_bias * users_bias[user_index])
            if update_items:
                items_bias[item_index] += learning_rate * (grad - K_bias * items_bias[item_index])

            # Update the users and items features vector
            for f in xrange(dim):
                cf = v[f,user_index]
                mf = u[f,item_index]
                # Main learned parameters
                if update_items:
                    u[f,item_index] += learning_rate * (grad * cf - K_items * mf)
                if update_users:
                    v[f,user_index] += learning_rate * (grad * mf - K_users * cf)
                
                
        rmse = sqrt(squared_error / nbr_ratings)
        epoch += 1
        
        if verbose:   
            print "EPOCHS: " + str(epoch)
            print "RMSE: " + str(rmse) + "\n"    
        

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)      
def train_linear_kernel(int nbr_epochs,
                        int dim,
                        double f_init,
                        double learning_rate,
                        double learning_rate_users,
                        double learning_rate_items,
                        double K_users,
                        double K_items,
                        double K_bias,
                        double overall_avg,
                        np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                        np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                        np.ndarray[np.int32_t, ndim=2, mode="c"] ratings_index,
                        np.ndarray[np.float64_t, ndim=1, mode="c"] ratings,
                        np.ndarray[np.float64_t, ndim=1, mode="c"] items_bias,
                        np.ndarray[np.float64_t, ndim=1, mode="c"] users_bias,
                        int update_users = 1,
                        int update_items = 1,
                        int verbose = 0):
    '''
    Main Loop for the Stochastic Gradient Descent training process for the Kernel Matrix Factorization
    method with a linear kernel
    '''
    
    cdef double rmse = 2.0
    cdef double rmse_last  = 0.0
    cdef double error = 0.0
    cdef double squared_error = 0.0
    cdef double cf = 0.0
    cdef double mf = 0.0
    cdef double rating = 0.0
    cdef double p = 0.0
    cdef double dot_prod, sig_dot = 0.0
    cdef double rating_range_size = 4.0
    cdef double min_rating = 1.0
    cdef double grad = 0.0

    cdef int epoch = 0
    cdef int user_index = 0
    cdef int item_index = 0
    cdef int f = 0
    cdef long i = 0
    cdef long nbr_ratings = 0
    
    nbr_ratings = ratings.shape[0]
    
    for epoch in xrange(nbr_epochs):
        squared_error = 0.0
        for i in xrange(nbr_ratings): 
            # Retrieves the rating
            user_index = ratings_index[i,0]
            item_index = ratings_index[i,1]
            rating = ratings[i]

            # Performs the estimation
            dot_prod = full_estimator(item_index, user_index, f, u, v, dim, 0.0, items_bias[item_index], users_bias[user_index])
            p  = dot_prod

            # Compute the error on the prediction and set the gradient accordingly 
            error = (rating - p);
            squared_error += error * error
            grad = error

            # Update the users and items bias
            users_bias[user_index] += learning_rate * (grad - K_bias * users_bias[user_index])
            items_bias[item_index] += learning_rate * (grad - K_bias * items_bias[item_index])

            # Update the users and items features vector
            for f in xrange(dim):
                cf = v[f,user_index]
                mf = u[f,item_index]
                # Main learned parameters
                if update_items:
                    u[f,item_index] += learning_rate * (grad * cf - K_items * mf)
                if update_users:
                    v[f,user_index] += learning_rate * (grad * mf - K_users * cf)
                
                
        rmse = sqrt(squared_error / nbr_ratings)
        epoch += 1
        
        if verbose:    
            print "RMSE: " + str(rmse) + "\n"    

