'''
    Cython module
    
    Created on October 19th, 2011
    
    @author: Martin Laprise
    
'''
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
import cython
import numpy as np
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
def als_wrmf_dense(int nbr_epochs,
             int dim,
             np.ndarray[np.float64_t, ndim=2, mode="c"] u,
             np.ndarray[np.float64_t, ndim=2, mode="c"] v,
             np.ndarray[np.float64_t, ndim=2] m,
             np.ndarray[np.float64_t, ndim=2] m_inv,             
             np.ndarray[np.int8_t, ndim=2, mode="c"] ratings,
             int nbr_users,
             int nbr_items,
             int c_pos = 1,
             double k = 0.015,
             int verbose = 0):
    
    '''
    Main loop for the Alternate least Square method that minimize the 
    Weighted Regularized Matrix Factorization
    '''
    #nbr_ratings = ratings.shape[0]

    #cdef np.ndarray[np.float64_t, ndim=2, mode="c"] m = np.zeros([dim, dim], dtype = np.float64)
    #cdef np.ndarray[np.float64_t, ndim=2, mode="c"] m_inv = np.zeros([dim, dim], dtype = np.float64)    
    #cdef np.ndarray[np.float64_t, ndim=2, mode="c"] m
    #cdef np.ndarray[np.float64_t, ndim=2, mode="c"] m_inv
    
    #cdef double* rating_cache = <double *> malloc(nbr_ratings * sizeof(double))    
    cdef double* HH = <double *> malloc(dim * dim * sizeof(double))
    cdef double* HC_minus_IH =  <double *> malloc(dim * dim * sizeof(double))
    cdef double* HCp = <double *> malloc(dim * sizeof(double))
    cdef double d = 0.0

    cdef int epoch = 0
    cdef int user_index = 0
    cdef int item_index = 0
    cdef int f, f1, f2 = 0

    cdef long i = 0
    cdef long j = 0
    cdef long nbr_ratings = 0
    
    for epoch in xrange(nbr_epochs):
        '''
        User pass
        '''
        print "Nbr. epoch: " + str(epoch)
        
        for f1 in xrange(dim):
            for f2 in xrange(dim):
                d = 0.0
                for i in xrange(nbr_items):
                    d += u[f1,i] * u[f2,i]
                HH[f1 + dim*f2] = d

        
        for j in xrange(nbr_users):
            '''
            Main Loop
            '''
            for f1 in xrange(dim):
                for f2 in xrange(dim): 
                    d = 0.0
                    for i in xrange(nbr_items):
                        if ratings[j,i] == 1:
                            d += u[f1,i] * u[f2,i] * c_pos
                    HC_minus_IH[f1 + dim * f2] = d
        

            for f in xrange(dim):
                d = 0.0
                for i in xrange(nbr_items):
                    if ratings[j,i] == 1:
                        d += u[f,i] * (1 + c_pos)
                HCp[f] = d
            

            for f1 in xrange(dim):
                for f2 in xrange(dim):
                    d = HH[f1 + dim*f2] + HC_minus_IH[f1 + dim * f2]
                    if (f1 == f2):
                        d += k
                    m[f1,f2] = d
            
            m_inv = np.linalg.inv(m)
            
            for f in xrange(dim):
                d = 0.0
                for f2 in xrange(dim): 
                    d += m_inv[f,f2] * HCp[f2]
                v[f,j] = d

        '''
        Item pass
        '''
        for f1 in xrange(dim):
            for f2 in xrange(dim):
                d = 0.0
                for i in xrange(nbr_users):
                    d += v[f1,i] * v[f2,i]
                HH[f1 + dim*f2] = d

        
        for j in xrange(nbr_items):
            '''
            Main Loop
            '''
            for f1 in xrange(dim):
                for f2 in xrange(dim): 
                    d = 0.0
                    for i in xrange(nbr_users):
                        if ratings[i,j] == 1:                        
                            d += v[f1,i] * v[f2,i] * c_pos
                    HC_minus_IH[f1 + dim * f2] = d
        

            for f in xrange(dim):
                d = 0.0
                for i in xrange(nbr_users):
                    if ratings[i,j] == 1:
                        d += v[f,i] * (1 + c_pos)
                HCp[f] = d
            

            for f1 in xrange(dim):
                for f2 in xrange(dim):
                    d = HH[f1 + dim*f2] + HC_minus_IH[f1 + dim * f2]
                    if (f1 == f2):
                        d += k
                    m[f1,f2] = d
            
            m_inv = np.linalg.inv(m)
           
            for f in xrange(dim):
                d = 0.0
                for f2 in xrange(dim): 
                    d += m_inv[f,f2] * HCp[f2];
                u[f,j] = d


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)      
def als_wrmf(int nbr_epochs,
             int dim,
             np.ndarray[np.float64_t, ndim=2, mode="c"] u,
             np.ndarray[np.float64_t, ndim=2, mode="c"] v,
             np.ndarray[np.float64_t, ndim=2] m,
             np.ndarray[np.float64_t, ndim=2] m_inv,             
             np.ndarray[np.int32_t, ndim=1, mode="c"] ratings_users_row,
             np.ndarray[np.int32_t, ndim=1, mode="c"] ratings_users_col,
             np.ndarray[np.int32_t, ndim=1, mode="c"] ratings_items_row,
             np.ndarray[np.int32_t, ndim=1, mode="c"] ratings_items_col,             
             int nbr_users,
             int nbr_items,
             int c_pos = 1,
             double k = 0.015,
             int verbose = 0):
    
    '''
    Main loop for the Alternate least Square method that minimize the 
    Weighted Regularized Matrix Factorization
    '''
    
    cdef double* HH = <double *> malloc(dim * dim * sizeof(double))
    cdef double* HC_minus_IH =  <double *> malloc(dim * dim * sizeof(double))
    cdef double* HCp = <double *> malloc(dim * sizeof(double))
    cdef double d = 0.0

    cdef int epoch = 0
    cdef int user_index = 0
    cdef int item_index = 0
    cdef int f, f1, f2 = 0

    cdef long i, j = 0
    cdef long start, span = 0
    cdef long nbr_ratings = 0
    cdef long nbr_active_users, nbr_active_items

    nbr_active_users = ratings_users_row.shape[0] - 1
    nbr_active_items = ratings_items_row.shape[0] - 1
    
    for epoch in xrange(nbr_epochs):
        '''
        User pass
        '''
        if verbose:
            print 'Epoch : ' + str(epoch)
        
        for f1 in xrange(dim):
            for f2 in xrange(dim):
                d = 0.0
                for i in xrange(nbr_items):
                    d += u[f1,i] * u[f2,i]
                HH[f1 + dim*f2] = d

        start = 0 
        for j in xrange(nbr_active_users):
            '''
            Main Loop
            '''
            start += ratings_users_row[j]
            span = ratings_users_row[j + 1]
            
            for f1 in xrange(dim):
                for f2 in xrange(dim): 
                    d = 0.0
                    for i in xrange(span):
                        item_index = ratings_users_col[start + i]
                        d += u[f1,item_index] * u[f2,item_index] * c_pos
                    HC_minus_IH[f1 + dim * f2] = d
        

            for f in xrange(dim):
                d = 0.0
                for i in xrange(span):
                    item_index = ratings_users_col[start + i]
                    d += u[f,item_index] * (1 + c_pos)
                HCp[f] = d
            

            for f1 in xrange(dim):
                for f2 in xrange(dim):
                    d = HH[f1 + dim*f2] + HC_minus_IH[f1 + dim * f2]
                    if (f1 == f2):
                        d += k
                    m[f1,f2] = d
            
            m_inv = np.linalg.inv(m)
            
            for f in xrange(dim):
                d = 0.0
                for f2 in xrange(dim): 
                    d += m_inv[f,f2] * HCp[f2]
                v[f,j] = d

        '''
        Item pass
        '''
        for f1 in xrange(dim):
            for f2 in xrange(dim):
                d = 0.0
                for i in xrange(nbr_users):
                    d += v[f1,i] * v[f2,i]
                HH[f1 + dim*f2] = d

        start = 0
        for j in xrange(nbr_active_items):
            '''
            Main Loop
            '''
            start += ratings_items_row[j]
            span = ratings_items_row[j + 1]
            
            for f1 in xrange(dim):
                for f2 in xrange(dim): 
                    d = 0.0
                    for i in xrange(span):
                        user_index = ratings_items_col[start + i]
                        d += v[f1,user_index] * v[f2,user_index] * c_pos
                    HC_minus_IH[f1 + dim * f2] = d
        

            for f in xrange(dim):
                d = 0.0
                for i in xrange(span):
                    user_index = ratings_items_col[start + i]
                    d += v[f,user_index] * (1 + c_pos)
                HCp[f] = d
            

            for f1 in xrange(dim):
                for f2 in xrange(dim):
                    d = HH[f1 + dim*f2] + HC_minus_IH[f1 + dim * f2]
                    if (f1 == f2):
                        d += k
                    m[f1,f2] = d
            
            m_inv = np.linalg.inv(m)
            
            for f in xrange(dim):
                d = 0.0
                for f2 in xrange(dim): 
                    d += m_inv[f,f2] * HCp[f2];
                u[f,j] = d


        
