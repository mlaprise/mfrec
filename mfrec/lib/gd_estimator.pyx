'''
    Cython module
    
    Estimator for the Gradient Descent
    
    There is two versions. A plain version using pure python list (gd_estimator)
    and a second one using numpy array. The numpy array version performs about the
    same as the pure list but without the numerical bug. The list-version currently
    have a bug. This Cython version (estimator_loop2) have a speedup of 800X over the plain prototype in
    pure-python (feature_training)
    
    Created on July 18th, 2011
    
    @author: Martin Laprise
    
'''
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
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


cdef double estimator_bias(int item_index,
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
    
    Evaluate and distribute the bias each time

    '''
    
    cdef double sum
    
    if cache > 0:
        sum = cache
    else:
        sum = overall_avg

    sum += u * v
    sum += (item_bias + user_bias) / dim

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
                        double f_init,
                        double cache = 0.0,
                        int trailing = 0,
                        double overall_avg = 1.0,
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
    sum = clamping(sum)
        
    if trailing == 1:
        sum += (dim - f - 1) * f_init * f_init
        sum = clamping(sum)
        
    return sum


@cython.boundscheck(False)
@cython.cdivision(True)  
cdef double full_estimator_implicit(int item_index,
                        int user_index,
                        int f,
                        np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                        np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                        np.ndarray[np.float64_t, ndim=2, mode="c"] y,
                        double feedback_norm,
                        int dim,
                        double f_init,
                        double cache = 0.0,
                        int trailing = 0,
                        double overall_avg = 1.0,
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
        sum += u[feature, item_index] * (v[feature, user_index] + feedback_norm * y[f,item_index]) 
    sum = clamping(sum)
        
    if trailing == 1:
        sum += (dim - f - 1) * f_init * f_init
        sum = clamping(sum)
        
    return sum



cdef predictor(int feature_index, 
               int user_index,
               np.ndarray[np.float64_t, ndim=2, mode="c"] u,
               np.ndarray[np.float64_t, ndim=2, mode="c"] v,
               int dim, 
               double feature_bias = 1.0):
    '''
    Predict the rating of unknown user-item pair
    '''

    cdef double sum = feature_bias
    
    for f in range(dim):
        sum += u[f,feature_index] * v[f,user_index]
        sum = clamping(sum)
        
    return sum


@cython.boundscheck(False)
@cython.cdivision(True)      
def estimator_loop(int min_epochs,
                    int max_epochs,
                    double min_improvement,
                    int dim,
                    double f_init,
                    double learning_rate,
                    double K,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] ratings_index,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] ratings,
                    int batch,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] rmse_hist,
                    int nbr_users,
                    int nbr_features,
                    int verbose = 0):
    '''
    Main Loop for the Gradient Descent method optimized with Cython

    Warning : This is not a python function
    This is ugly but this is about 800X faster than the pure-python version...
    '''
    
    cdef double rmse = 2.0
    cdef double rmse_last  = 0.0
    cdef double error = 0.0
    cdef double squared_error = 0.0
    cdef double cf = 0.0
    cdef double mf = 0.0
    cdef double rating = 0.0
    cdef double p = 0.0
    cdef double improvement = 0.0
   
    cdef int epoch = 0
    cdef int user_index = 0
    cdef int feature_index = 0
    cdef int f = 0

    cdef long i = 0
    cdef long nbr_ratings = 0
    cdef long cache_size = nbr_features * nbr_users
    
    cdef double* rating_cache = <double *> malloc(cache_size * sizeof(double))
    
    for i in xrange(cache_size):
        rating_cache[i] = 0.0

    nbr_ratings = ratings.shape[0]
        
    for f in xrange(dim):
        if verbose:
            print "Training the feature " + str(f)
        epoch = 0
        
        while ( (epoch < min_epochs or improvement >= min_improvement) and epoch < max_epochs):
            squared_error = 0.0
            rmse_last = rmse
            for i in xrange(nbr_ratings):    
                user_index = ratings_index[i,0]
                feature_index = ratings_index[i,1]
                rating = ratings[i]

                p = estimator(feature_index, user_index, f, u[f,feature_index], v[f,user_index], dim, f_init,
                              rating_cache[user_index + feature_index * nbr_users], 1)
                
                error = (1.0 * rating - p);
                squared_error += error * error;
                
                cf = v[f,user_index]
                mf = u[f,feature_index]
                
                v[f,user_index] += learning_rate * (error * mf - K * cf)
                u[f,feature_index] += learning_rate * (error * cf - K * mf)
            
            rmse = sqrt(squared_error / nbr_ratings)
            rmse_hist[epoch + f * max_epochs + batch * max_epochs * dim] = rmse
            improvement = rmse_last - rmse

            epoch += 1
        
        if verbose:
            print "Nbr. epoch: " + str(epoch)
            print "RMSE: " + str(rmse) + "\n"
        
        # Store the ratings in cache
        for i in xrange(nbr_ratings):    
            user_index = ratings_index[i,0]
            feature_index = ratings_index[i,1]
            rating = ratings[i]
            
            rating_cache[user_index + feature_index * nbr_users] = estimator(feature_index, user_index, f, u[f,feature_index], v[f,user_index],
                dim, f_init, rating_cache[user_index + feature_index * nbr_users], 0)
    
    free(rating_cache)


@cython.boundscheck(False)
@cython.cdivision(True)      
def estimator_loop2(int min_epochs,
                    int max_epochs,
                    double min_improvement,
                    int dim,
                    double f_init,
                    double learning_rate,
                    double K,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] ratings_index,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] ratings,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] features_bias,
                    int nbr_users,
                    int nbr_features,
                    int verbose = 0):
    '''
    Main Loop for the Gradient Descent method optimized with Cython

    Warning : This is not a python function
    This is ugly but this is about 800X faster than the pure-python version...
    '''
    
    cdef double rmse = 2.0
    cdef double rmse_last  = 0.0
    cdef double error = 0.0
    cdef double squared_error = 0.0
    cdef double cf = 0.0
    cdef double mf = 0.0
    cdef double rating = 0.0
    cdef double p = 0.0
   
    cdef int epoch = 0
    cdef int user_index = 0
    cdef int feature_index = 0
    cdef int f = 0

    cdef long i = 0
    cdef long nbr_ratings = 0
    cdef long cache_size = nbr_features * nbr_users
    
    cdef double* rating_cache = <double *> malloc(cache_size * sizeof(double))
    
    for i in xrange(cache_size):
        rating_cache[i] = 0.0

    nbr_ratings = ratings.shape[0]
        
    for f in xrange(dim):
        if verbose:
            print "Training the feature " + str(f)
        epoch = 0
        
        while (epoch < min_epochs or rmse <= rmse_last - min_improvement):
            squared_error = 0.0
            rmse_last = rmse
            for i in xrange(nbr_ratings):    
                user_index = ratings_index[i,0]
                feature_index = ratings_index[i,1]
                rating = ratings[i]

                p = estimator(feature_index, user_index, f, u[f,feature_index], v[f,user_index], dim, f_init,
                              rating_cache[user_index + feature_index * nbr_users], 1)
                
                error = (1.0 * rating - p);
                squared_error += error * error;
                
                cf = v[f,user_index]
                mf = u[f,feature_index]
                
                v[f,user_index] += learning_rate * (error * mf - K * cf)
                u[f,feature_index] += learning_rate * (error * cf - K * mf)
                
            rmse = sqrt(squared_error / nbr_ratings)
            epoch += 1
        
        if verbose:    
            print "RMSE: " + str(rmse) + "\n"    
        
        # Store the ratings in cache
        for i in xrange(nbr_ratings):    
            user_index = ratings_index[i,0]
            feature_index = ratings_index[i,1]
            rating = ratings[i]
            
            rating_cache[user_index + feature_index * nbr_users] = estimator(feature_index, user_index, f, u[f,feature_index], v[f,user_index],
                dim, f_init, rating_cache[user_index + feature_index * nbr_users], 0)
    
    free(rating_cache)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)      
def estimator_loop_with_learned_bias(int min_epochs,
                    int max_epochs,
                    double min_improvement,
                    int dim,
                    double f_init,
                    double learning_rate,
                    double learning_rate_users,
                    double learning_rate_items,
                    double K_feature,
                    double K_bias,
                    double overall_avg,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] ratings_index,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] ratings,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] items_bias,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] users_bias,
                    int nbr_users,
                    int nbr_items,
                    int verbose = 0,
                    int learning_mode = 0):
    '''
    Main Loop for the Gradient Descent method optimized with Cython

    Warning : This is not a python function
    This is ugly but this is about 800X faster than the pure-python version...
    '''
    
    cdef double rmse = 2.0
    cdef double rmse_last  = 0.0
    cdef double error = 0.0
    cdef double squared_error = 0.0
    cdef double cf = 0.0
    cdef double mf = 0.0
    cdef double rating = 0.0
    cdef double p = 0.0
   
    cdef int epoch = 0
    cdef int user_index = 0
    cdef int item_index = 0
    cdef int f = 0

    cdef long i = 0
    cdef long nbr_ratings = 0
    cdef long cache_size = nbr_items * nbr_users
    
    nbr_ratings = ratings.shape[0]
    

    for f in xrange(dim):
        if verbose:
            print "Training the feature " + str(f)
        epoch = 0
        
        while (epoch < min_epochs or rmse <= rmse_last - min_improvement):
            squared_error = 0.0
            rmse_last = rmse
            for i in xrange(nbr_ratings):    
                user_index = ratings_index[i,0]
                item_index = ratings_index[i,1]
                rating = ratings[i]
    
                p = full_estimator(item_index, user_index, f, u, v, dim, f_init,
                              0, 1, overall_avg, items_bias[item_index], users_bias[user_index])
 
                error = (rating - p);
                squared_error += error * error;
                
                cf = v[f,user_index]
                mf = u[f,item_index]
              
                # Learned bias update
                users_bias[user_index] += learning_rate_users * (error - K_bias * users_bias[user_index])
                items_bias[item_index] += learning_rate_items * (error - K_bias * items_bias[item_index])
                # Learned feature update
                u[f,item_index] += learning_rate * (error * cf - K_feature * mf)
                v[f,user_index] += learning_rate * (error * mf - K_feature * cf)
                
            rmse = sqrt(squared_error / nbr_ratings)
            epoch += 1
        
        if verbose:    
            print "RMSE: " + str(rmse) + "\n"    


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)      
def estimator_loop_with_bias(int min_epochs,
                    int max_epochs,
                    double min_improvement,
                    int dim,
                    double f_init,
                    double learning_rate,
                    double learning_rate_users,
                    double learning_rate_items,
                    double K,
                    double overall_avg,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] ratings_index,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] ratings,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] items_bias,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] users_bias,
                    int nbr_users,
                    int nbr_items,
                    int verbose = 0):
    '''
    Main Loop for the Gradient Descent method optimized with Cython

    Warning : This is not a python function
    This is ugly but this is about 800X faster than the pure-python version...
    '''
    
    cdef double rmse = 2.0
    cdef double rmse_last  = 0.0
    cdef double error = 0.0
    cdef double squared_error = 0.0
    cdef double cf = 0.0
    cdef double mf = 0.0
    cdef double rating = 0.0
    cdef double p = 0.0
   
    cdef int epoch = 0
    cdef int user_index = 0
    cdef int item_index = 0
    cdef int f = 0

    cdef long i = 0
    cdef long nbr_ratings = 0
    cdef long cache_size = nbr_items * nbr_users
    
    nbr_ratings = ratings.shape[0]
    
    cdef double* rating_cache = <double *> malloc(nbr_ratings * sizeof(double))    
     
    for i in xrange(nbr_ratings):
        rating_cache[i] = 0.0

    for f in xrange(dim):
        if verbose:
            print "Training the feature " + str(f)
        epoch = 0
        
        while (epoch < min_epochs or rmse <= rmse_last - min_improvement):
            squared_error = 0.0
            rmse_last = rmse
            for i in xrange(nbr_ratings):    
                user_index = ratings_index[i,0]
                item_index = ratings_index[i,1]
                rating = ratings[i]
    
                p = estimator(item_index, user_index, f, u[f,item_index], v[f,user_index], dim, f_init,
                              rating_cache[i], 1, overall_avg, items_bias[item_index], users_bias[user_index])

                error = (rating - p);
                squared_error += error * error;
                
                cf = v[f,user_index]
                mf = u[f,item_index]
               
                # Main learned parameters
                u[f,item_index] += learning_rate * (error * cf - K * mf)
                v[f,user_index] += learning_rate * (error * mf - K * cf)
                
                
            rmse = sqrt(squared_error / nbr_ratings)
            epoch += 1
        
        if verbose:    
            print "RMSE: " + str(rmse) + "\n"    
        
        # Store the ratings in cache
        for i in xrange(nbr_ratings):    
            user_index = ratings_index[i,0]
            item_index = ratings_index[i,1]
            rating = ratings[i]
            
            rating_cache[i] = estimator(item_index, user_index, f, u[f,item_index], v[f,user_index],
                dim, f_init, rating_cache[i], 0, overall_avg, items_bias[item_index], users_bias[user_index])

    free(rating_cache)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)      
def estimator_loop_with_bias_dev(int min_epochs,
                    int max_epochs,
                    double min_improvement,
                    int dim,
                    double f_init,
                    double learning_rate,
                    double learning_rate_users,
                    double learning_rate_items,
                    double K,
                    double overall_avg,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] ratings_index,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] ratings,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] items_bias,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] users_bias,
                    int nbr_users,
                    int nbr_items,
                    int update_users = 1,
                    int update_items = 1,
                    int verbose = 0):
    '''
    Main Loop for the Gradient Descent method optimized with Cython

    Warning : This is not a python function
    This is ugly but this is about 800X faster than the pure-python version...
    '''
    
    cdef double rmse = 2.0
    cdef double rmse_last  = 0.0
    cdef double error = 0.0
    cdef double squared_error = 0.0
    cdef double cf = 0.0
    cdef double mf = 0.0
    cdef double rating = 0.0
    cdef double p = 0.0
   
    cdef int epoch = 0
    cdef int user_index = 0
    cdef int item_index = 0
    cdef int f = 0

    cdef long i = 0
    cdef long nbr_ratings = 0
    cdef long cache_size = nbr_items * nbr_users
    
    nbr_ratings = ratings.shape[0]
    
    cdef double* rating_cache = <double *> malloc(nbr_ratings * sizeof(double))    
     
    for i in xrange(nbr_ratings):
        rating_cache[i] = 0.0

    for f in xrange(dim):
        if verbose:
            print "Training the feature " + str(f)
        epoch = 0
        
        while (epoch < min_epochs or rmse <= rmse_last - min_improvement):
            squared_error = 0.0
            rmse_last = rmse
            for i in xrange(nbr_ratings):    
                user_index = ratings_index[i,0]
                item_index = ratings_index[i,1]
                rating = ratings[i]
    
                p = estimator(item_index, user_index, f, u[f,item_index], v[f,user_index], dim, f_init,
                              rating_cache[i], 1, overall_avg, items_bias[item_index], users_bias[user_index])

                error = (rating - p);
                squared_error += error * error;
                
                cf = v[f,user_index]
                mf = u[f,item_index]
               
                # Main learned parameters
                if update_items:
                    u[f,item_index] += learning_rate * (error * cf - K * mf)
                if update_users:
                    v[f,user_index] += learning_rate * (error * mf - K * cf)
                
                
            rmse = sqrt(squared_error / nbr_ratings)
            epoch += 1
        
        if verbose:    
            print "RMSE: " + str(rmse) + "\n"    
        
        # Store the ratings in cache
        for i in xrange(nbr_ratings):    
            user_index = ratings_index[i,0]
            item_index = ratings_index[i,1]
            rating = ratings[i]
            
            rating_cache[i] = estimator(item_index, user_index, f, u[f,item_index], v[f,user_index],
                dim, f_init, rating_cache[i], 0, overall_avg, items_bias[item_index], users_bias[user_index])

    free(rating_cache)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)      
def estimator_loop_without_bias(int min_epochs,
                    int max_epochs,
                    double min_improvement,
                    int dim,
                    double f_init,
                    double learning_rate,
                    double K,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] ratings_index,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] ratings,
                    int nbr_users,
                    int nbr_items,
                    int verbose = 0):
    '''
    Main Loop for the Gradient Descent method optimized with Cython

    Warning : This is not a python function
    This is ugly but this is about 800X faster than the pure-python version...
    '''
    
    cdef double rmse = 2.0
    cdef double rmse_last  = 0.0
    cdef double error = 0.0
    cdef double squared_error = 0.0
    cdef double cf = 0.0
    cdef double mf = 0.0
    cdef double rating = 0.0
    cdef double p = 0.0
   
    cdef int epoch = 0
    cdef int user_index = 0
    cdef int item_index = 0
    cdef int f = 0

    cdef long i = 0
    cdef long nbr_ratings = 0
    cdef long cache_size = nbr_items * nbr_users
    
    nbr_ratings = ratings.shape[0]
    
    cdef double* rating_cache = <double *> malloc(nbr_ratings * sizeof(double))    
     
    for i in xrange(nbr_ratings):
        rating_cache[i] = 0.0

    for f in xrange(dim):
        if verbose:
            print "Training the feature " + str(f)
        epoch = 0
        
        while (epoch < min_epochs or rmse <= rmse_last - min_improvement):
            squared_error = 0.0
            rmse_last = rmse
            for i in xrange(nbr_ratings):    
                user_index = ratings_index[i,0]
                item_index = ratings_index[i,1]
                rating = ratings[i]
                
                p = estimator(item_index, user_index, f, u[f,item_index], v[f,user_index], dim, f_init,
                              rating_cache[i], 1)
                #p = full_estimator(item_index, user_index, f, u, v, dim, f_init, rating_cache[i], 1)
                
                error = (1.0 * rating - p);
                squared_error += error * error;
                
                cf = v[f,user_index]
                mf = u[f,item_index]
                
                u[f,item_index] += learning_rate * (error * cf - K * mf)
                v[f,user_index] += learning_rate * (error * mf - K * cf)
                
                
            rmse = sqrt(squared_error / nbr_ratings)
            epoch += 1
        
        if verbose:    
            print "RMSE: " + str(rmse) + "\n"    
        
        # Store the ratings in cache
        for i in xrange(nbr_ratings):    
            user_index = ratings_index[i,0]
            item_index = ratings_index[i,1]
            rating = ratings[i]
            
            rating_cache[i] = estimator(item_index, user_index, f, u[f,item_index], v[f,user_index],
                dim, f_init, rating_cache[i], 0)
            
    free(rating_cache)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)      
def estimator_loop_with_implicit_feedback(int min_epochs,
                    int max_epochs,
                    double min_improvement,
                    int dim,
                    double f_init,
                    double learning_rate,
                    double learning_rate_users,
                    double learning_rate_items,
                    double K,
                    double overall_avg,                    
                    np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] y,                    
                    np.ndarray[np.int32_t, ndim=2, mode="c"] ratings_index,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] ratings,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] rated,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] rated_hash,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] items_bias,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] users_bias,                    
                    int nbr_users,
                    int nbr_items,
                    int verbose = 0):
    '''
    Main Loop for the regularized matrix factorization using a stochastic gradient descent method

    '''
    
    cdef double rmse = 2.0
    cdef double rmse_last  = 0.0
    cdef double error = 0.0
    cdef double squared_error = 0.0
    cdef double cf = 0.0
    cdef double mf = 0.0
    cdef double rating = 0.0
    cdef double p = 0.0
    cdef double feedback_norm = 0.0
    cdef double feedback_sum = 0.0
    cdef double v_implicit = 0.0
   
    cdef int epoch = 0
    cdef int f = 0
   
    cdef long nbr_rated = 0
    cdef long sum_j = 0 
    cdef long user_index = 0
    cdef long item_index = 0
    cdef long i = 0
    cdef long nbr_ratings = 0
    cdef long cache_size = nbr_items * nbr_users
    
    nbr_ratings = ratings.shape[0]
    
    cdef double* rating_cache = <double *> malloc(nbr_ratings * sizeof(double))    
     
    for i in xrange(nbr_ratings):
        rating_cache[i] = 0.0

    for f in xrange(dim):
        if verbose:
            print "Training the feature " + str(f)
        epoch = 0
        
        while (epoch < min_epochs or rmse <= rmse_last - min_improvement):
            squared_error = 0.0
            rmse_last = rmse
            for i in xrange(nbr_ratings):    
                user_index = ratings_index[i,0]
                item_index = ratings_index[i,1]
                rating = ratings[i]

                nbr_rated = rated_hash[user_index,1]                
                feedback_norm = (1.0 / float(sqrt(nbr_rated)))
                v_implicit = v[f,user_index] + (feedback_norm * y[f,item_index])
                
                p = estimator(item_index, user_index, f, u[f,item_index], v_implicit, dim, f_init,
                              rating_cache[i], 1, overall_avg, items_bias[item_index], users_bias[user_index])
                
                error = (1.0 * rating - p);
                squared_error += error * error;

                cf = v[f,user_index]
                
                # Implicit feedback
                feedback_sum = 0.0                
                for i in xrange(nbr_rated):
                    sum_j = int(rated[rated_hash[user_index,0] + i,1])
                    feedback_sum += y[f,sum_j]
                
                feedback_sum = y[f,item_index]
                mf = u[f,item_index]
                
                u[f,item_index] += learning_rate * (error * (cf + feedback_norm*feedback_sum) - K * mf)
                v[f,user_index] += learning_rate * (error * mf - K * cf)
                
                for i in xrange(nbr_rated):
                    sum_j = int(rated[rated_hash[user_index,0] + i,1])
                    y[f,sum_j] += learning_rate * (error * feedback_norm * mf - K * y[f,sum_j])
                
            rmse = sqrt(squared_error / nbr_ratings)
            epoch += 1
        
        if verbose:    
            print "RMSE: " + str(rmse) + "\n"
        
        # Store the ratings in cache
        for i in xrange(nbr_ratings):    
            user_index = ratings_index[i,0]
            item_index = ratings_index[i,1]
            rating = ratings[i]
            
            rating_cache[i] = estimator(item_index, user_index, f, u[f,item_index], v[f,user_index],
                dim, f_init, rating_cache[i], 0, overall_avg, items_bias[item_index], users_bias[user_index])
            
    free(rating_cache)


@cython.boundscheck(False)
@cython.cdivision(True)      
def estimator_subloop(int f,
                    int epochs,
                    double min_improvement,
                    int dim,
                    double f_init,
                    double learning_rate,
                    double K,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                    np.ndarray[np.int32_t, ndim=2, mode="c"] ratings_index,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] ratings,
                    np.ndarray[np.float64_t, ndim=1, mode="c"] rating_cache,
                    int nbr_users,
                    int nbr_features,
                    int verbose = 0):
    '''

    Sub Loop for the Gradient Descent method optimized with Cython
    
    '''
    
    cdef double error = 0.0
    cdef double squared_error = 0.0
    cdef double rmse = 0.0
    cdef double cf = 0.0
    cdef double mf = 0.0
    cdef double rating = 0.0
    cdef double p = 0.0
   
    cdef int user_index = 0
    cdef int feature_index = 0

    cdef long i = 0
    cdef long nbr_ratings = 0
    cdef long cache_size = nbr_features * nbr_users
     
    nbr_ratings = ratings.shape[0]
             
    squared_error = 0.0

    for i in xrange(nbr_ratings):    
        user_index = ratings_index[i,0]
        feature_index = ratings_index[i,1]
        rating = ratings[i]

        p = estimator(feature_index, user_index, f, u[f,feature_index], v[f,user_index], dim, f_init,
                      rating_cache[user_index + feature_index * nbr_users], 1)
        
        error = (1.0 * rating - p);
        squared_error += error * error;
        
        cf = v[f,user_index]
        mf = u[f,feature_index]
        
        v[f,user_index] += learning_rate * (error * mf - K * cf)
        u[f,feature_index] += learning_rate * (error * cf - K * mf)
        
    rmse = sqrt(squared_error / nbr_ratings)
        
    return rmse


@cython.boundscheck(False)
@cython.cdivision(True)      
def predictor_subloop(int f,
                      int epochs,
                      int dim,
                      double f_init,
                      np.ndarray[np.float64_t, ndim=2, mode="c"] u,
                      np.ndarray[np.float64_t, ndim=2, mode="c"] v,
                      np.ndarray[np.int32_t, ndim=2, mode="c"] ratings_index,
                      np.ndarray[np.float64_t, ndim=1, mode="c"] ratings,
                      np.ndarray[np.float64_t, ndim=1, mode="c"] rating_cache,
                      int nbr_users,
                      int nbr_features):

    cdef double rating = 0.0
   
    cdef int user_index = 0
    cdef int feature_index = 0

    cdef long i = 0
    cdef long nbr_ratings = 0
     
    nbr_ratings = ratings.shape[0]

    for i in xrange(nbr_ratings):    
        user_index = ratings_index[i,0]
        feature_index = ratings_index[i,1]
        rating = ratings[i]
        
        rating_cache[user_index + feature_index * nbr_users] = estimator(feature_index, user_index, f, u[f,feature_index], v[f,user_index],
            dim, f_init, rating_cache[user_index + feature_index * nbr_users], 0)
      
