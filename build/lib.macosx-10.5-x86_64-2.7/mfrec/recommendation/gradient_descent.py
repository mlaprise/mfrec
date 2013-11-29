'''
.. container:: creation-info

    Created on July 11th, 2010

    @author Martin Laprise

Recommender based on a Stochastic Gradient Descent approach

'''
import math
from operator import itemgetter
import itertools

import numpy as np
import scipy.sparse
from scipy.sparse import find
from scipy.sparse import lil_matrix, csc_matrix, find

from mfrec.config import base
from mfrec.recommendation.base import BaseRecommender
from mfrec.recommendation.mf import MFRecommender
from mfrec.recommendation.metrics import test_predict_rating
from mfrec.lib.machinelearning.gd_estimator import estimator_loop, estimator_loop2, estimator_loop_with_bias, estimator_loop_with_bias_dev, \
                                                           estimator_subloop, predictor_subloop, estimator_loop_without_bias, \
                                                           estimator_loop_with_implicit_feedback, estimator_loop_with_learned_bias

class GDRecommender(MFRecommender):
    '''
    
    Recommender based on a regularized singular value decomposition minimize with a 
    Stochastic Gradient Descent method
    
    This recommender use a Stochastic Gradient Descent for computing each features. When the matrix is not sparse
    this method is equivalent to a Singular Value Decomposition. Each feature is compute using the Gradient Descent
    on the error a single rating (pseudo-stochastic). The over-fitting is control using a regularization term (K)
    The optimal parameters for the training depends on the dataset.

    Complexity : The SGD method scale like O(kni) where n ist the number of users, k the number of epoch, and i is
                 the number average number of non-zero attributes per item.
    
    Training on 10M ratings
        * model: 40 features
        * dataset : MovieLens Dataset 10M (69879 users and 10678 items)
        * time: 49 minutes 
        * Machine: Core i5 2.3ghz with 4GB

    reference:
    
    * Simon Funk's article
      http://sifter.org/~simon/journal/20061211.html
    
    * Yehuda Koren, Factorization meets the neighborhood: a multifaceted collaborative filtering model, Proceeding of the
      14th ACM SIGKDD international conference on Knowledge discovery and data mining, August 24-27, 2008, Las Vegas, Nevada,
      USA  [doi>10.1145/1401890.1401944]
      http://www.commendo.at/references/files/kdd08.pdf
    
    '''


    PARAMETERS_INDEX = {'min_epochs' : 'min_epochs',
                        'max_epochs' : 'max_epochs',
                        'min_improvement' : 'min_improvement',
                        'feature_init' : 'feature_init',
                        'learning_rate' : 'learning_rate',
                        'learning_rate_users' : 'learning_rate_users',
                        'learning_rate_items' : 'learning_rate_items',
                        'regularization_model' : 'K',
                        'regularization_users_bias' : 'K2',
                        'regularization_items_bias' : 'K3',
                        'nbr_features' : 'dimensionality'}


    def __init__(self,  nbr_users = 4, nbr_items = 6, parameters = False, filename = False):
        MFRecommender.__init__(self, nbr_users, nbr_items, parameters)

        # Initialize the training parameters with the default value
        self.min_epochs = 275
        self.max_epochs = 275
        self.min_improvement = 0.0001
        self.feature_init = 0.1
        self.learning_rate = 0.001
        self.learning_rate_users = 0.001
        self.learning_rate_items = 0.001
        self.K = 0.05
        self.K2 = 0.01
        self.K3 = 0.01
        self.dimensionality = 40

        if parameters:
            self.set_parameters(parameters)
            
        self.batch_size = 10
        self.rmse_history = np.zeros(self.max_epochs)
        self.rating_cache = None
        self.nbr_ratings = None
        self.global_avg = None
        self.mongodb_iterator = None
        self.components_mean = None
        self.N = None
        self.items_feedback = None
        self.feedback_rated = None
        self.feedback_hash = None
    
    
    def __repr__(self):
        string = 'Gradient Descent based Recommendation Engine\n'
        string += 'Number of users: ' + str(self.nbr_users) + '\n'
        string += 'Number of items: ' + str(self.nbr_items) + '\n'
        string += 'Dimensionality: ' + str(self.dimensionality) + '\n'
        return string
        
    
    def set_ratings_iterator(self, iterator): 
        self.mongodb_iterator = iterator


    def get_rmse_history(self):
        index = self.rmse_history.nonzero()
        return self.rmse_history[index]

    
    def get_nbr_ratings(self):
        '''
        Return the number of non-zero item in the relationshion_matrix (nbr. of ratings).
        '''
        return find(self.relationship_matrix)[1].shape[0]
    

    def feature_training_prototype_p(self, verbose = False, randomize = False):
        '''
        Compute each features using a Gradient Descent approach. This method is the pure python version of the 
        feature_training() method. It should be only used as a dev tools (very slow). 
        
        This particular version is a prototype for a parallel version of the feature_training_prototype using
        the approach described in Zinkevich et al (Parallelized Stochastic Gradient Descent, 2011). The sections
        called Machine 1 and Machine 2 can be move on a different process, thread or machine with the respective
        subset of data. The only communication between machines occurs at the end, where we take the average of
        each model parameters computed on each machine. It's pretty simple but Zinkevich et al. proved the
        convergence of this method.
        '''
        
        rmse = 2.0
        rmse_last = 2.0
        self.rating_cache = np.zeros(self.relationship_matrix.get_shape())
        
        self.svd_v = np.zeros([self.dimensionality, self.nbr_users]) + self.feature_init
        self.svd_u = np.zeros([self.dimensionality, self.nbr_items]) + self.feature_init 

        nbr_ratings = self.get_nbr_ratings()
        ratings_index, ratings = self.get_ratings(randomize_order = False)
        batch_index = np.arange(nbr_ratings)
        nbr_batch = 2
        cuts = np.linspace(0, nbr_ratings, nbr_batch + 1).astype(int)

        for f in np.arange(self.dimensionality):
            if verbose:
                print "Training the feature " + str(f)
            epoch = 0
            
            while (epoch < self.min_epochs or rmse <= rmse_last - self.min_improvement):
                squared_error = 0.0
                rmse_last = rmse
              
                self.w1a = self.svd_v[f,:] 
                self.w2a = self.svd_u[f,:] 
                self.w1b = self.svd_v[f,:] 
                self.w2b = self.svd_u[f,:] 

                # Machine 1                               
                for i in batch_index[cuts[0]:cuts[1]]:
                    user_index = ratings_index[i,0]
                    feature_index = ratings_index[i,1]
                    rating = ratings[i]

                    p = self.estimate_rating(feature_index, user_index, f, self.rating_cache[user_index, feature_index], trailing = 1)
                    error = (1.0 * rating - p);
                    squared_error += error * error;
                    
                    cf = self.svd_v[f,user_index]
                    mf = self.svd_u[f,feature_index]
                    
                    self.w1a[user_index] += self.learning_rate * (error * mf - self.K * cf)
                    self.w2a[feature_index] += self.learning_rate * (error * cf - self.K * mf)

                squared_error = 0.0
                rmse_last = rmse
                
                # Machine 2                
                for i in batch_index[cuts[1]:cuts[2]]:
                    user_index = ratings_index[i,0]
                    feature_index = ratings_index[i,1]
                    rating = ratings[i]

                    p = self.estimate_rating(feature_index, user_index, f, self.rating_cache[user_index, feature_index], trailing = 1)
                    error = (1.0 * rating - p);
                    squared_error += error * error;
                    
                    cf = self.svd_v[f,user_index]
                    mf = self.svd_u[f,feature_index]
                    
                    self.w1b[user_index] += self.learning_rate * (error * mf - self.K * cf)
                    self.w2b[feature_index] += self.learning_rate * (error * cf - self.K * mf)
                
                self.svd_v[f,:] = (self.w1a + self.w1b) / 2
                self.svd_u[f,:] = (self.w2a + self.w2b) / 2
                
                rmse = np.sqrt(squared_error / nbr_ratings)
                
                if verbose:
                    print "Epoch: " + str(epoch)
                    print "RMSE: " + str(rmse) + "\n"
                
                epoch += 1
                
            for user_index, feature_index, rating in self.ratings_iterator():
                self.rating_cache[user_index, feature_index] = self.estimate_rating(feature_index, user_index, f, self.rating_cache[user_index, feature_index], trailing = 0)
 

    def feature_training_prototype(self, verbose = False, randomize = False):
        '''
        Compute each features using a Gradient Descent approach. This method is the pure python version of the 
        feature_training() method. It should be only used as a dev tools (very slow). 
        '''
        
        rmse = 2.0
        rmse_last = 2.0
        self.rating_cache = np.zeros(self.relationship_matrix.get_shape())
        
        self.svd_v = np.zeros([self.dimensionality, self.nbr_users]) + self.feature_init
        self.svd_u = np.zeros([self.dimensionality, self.nbr_items]) + self.feature_init 

        nbr_ratings = self.get_nbr_ratings()
        
        for f in np.arange(self.dimensionality):
            if verbose:
                print "Training the feature " + str(f)
            epoch = 0
            
            while (epoch < self.min_epochs or rmse <= rmse_last - self.min_improvement):
                squared_error = 0.0
                rmse_last = rmse
                for user_index, feature_index, rating in self.ratings_iterator():
                    p = self.estimate_rating(feature_index, user_index, f, self.rating_cache[user_index, feature_index], trailing = 1)
                    error = (1.0 * rating - p);
                    squared_error += error * error;
                    
                    cf = self.svd_v[f,user_index]
                    mf = self.svd_u[f,feature_index]
                    
                    self.svd_v[f,user_index] += self.learning_rate * (error * mf - self.K * cf)
                    self.svd_u[f,feature_index] += self.learning_rate * (error * cf - self.K * mf)
                    
                rmse = np.sqrt(squared_error / nbr_ratings)
                
                if verbose:
                    print "Epoch: " + str(epoch)
                    print "RMSE: " + str(rmse) + "\n"
                
                epoch += 1
                
            for user_index, feature_index, rating in self.ratings_iterator():
                self.rating_cache[user_index, feature_index] = self.estimate_rating(feature_index, user_index, f, self.rating_cache[user_index, feature_index], trailing = 0)
  
        

    def initialize_rated_feedback(self):
        self.N = self.relationship_matrix.astype(bool)
        self.feedback_rated, self.feedback_hash = self.get_feedback()
        
    
    def N_iterator(self):
        cx = self.N.tocoo()
        return itertools.izip(cx.row, cx.col)


    def get_feedback(self):
        
        nbr_ratings = find(self.N)[2].shape[0]
        ratings_id = np.zeros([nbr_ratings,2], dtype = np.int32)
        ratings_hash = np.zeros([self.nbr_users,2], dtype = np.int32)

        for i, (user_index, feature_index) in enumerate(self.N_iterator()):
            ratings_id[i] = [int(user_index), int(feature_index)]
        
        index = np.arange(nbr_ratings)

        # Add sort here
        
        # We assume ratings_id is sorted
        for u in range(self.nbr_users):
            position = np.where(ratings_id[:,0] == u)[0]
            try:
                seek = position[0]
                span = len(position)
            except:
                seek = -1
                span = -1
            
            ratings_hash[u] = [seek, span]
            
        return ratings_id[index], ratings_hash


    def feature_training2(self, initialize_model = True, verbose = False):
        '''
        Compute each features using a Gradient Descent approach
        This version call the Cython estimator_loop2() function
        '''
        
        rmse = 2.0
        
        # Initialize the model with previous results if available
        if initialize_model:
            self.svd_v = np.zeros([self.dimensionality, self.nbr_users]) + self.feature_init
            self.svd_u = np.zeros([self.dimensionality, self.nbr_items]) + self.feature_init
        
        nbr_ratings = find(self.relationship_matrix)[2].shape[0]
        ratings_cache = np.zeros(self.nbr_users * self.nbr_items, dtype = np.float64)
        
        ratings_index, ratings = self.get_ratings()
        
        for f in range(self.dimensionality):
            epoch = 0
            
            while (epoch < self.min_epochs or rmse <= rmse_last - self.min_improvement):
                rmse_last = rmse
                rmse = estimator_subloop(f, epoch, self.min_improvement, self.dimensionality, self.feature_init, self.learning_rate,
                        self.K, self.svd_u, self.svd_v, ratings_index, ratings, ratings_cache, self.nbr_users, self.nbr_items, int(verbose))
                
                epoch += 1
                
            predictor_subloop(f, epoch, self.dimensionality, self.feature_init, self.svd_u, self.svd_v, ratings_index, ratings, ratings_cache,
                                self.nbr_users, self.nbr_items)
          
   
    def feature_training_batch(self, batch_size = 10000, probeset = None, verbose = False):
        ratings_index, ratings = self.get_ratings(randomize_order = False)
        
        nbr_ratings = self.get_nbr_ratings()
        nbr_batchs = nbr_ratings / batch_size
        bounds = np.linspace(0, nbr_ratings, nbr_batchs + 1).astype(int)
        rmse_hists = np.zeros([nbr_batchs, self.max_epochs * self.dimensionality])
        rmse_probe = np.zeros(nbr_batchs)

        rmse_hists[0] = self.feature_training_online(ratings_index[bounds[0]:bounds[1]], ratings[bounds[0]:bounds[1]], verbose = verbose)
        
        if probeset != None:
            rmse_probe[0] = test_predict_rating(self, probeset, nbr_samples = 20000)

        for b in xrange(nbr_batchs):
            print 'Training batch ' + str(b)
            self.feature_training_online_prototype(ratings_index[bounds[b]:bounds[b+1]],
                                         ratings[bounds[b]:bounds[b+1]],
                                         initialize_cache = True,
                                         initialize_model = False,
                                         verbose = verbose)
            if probeset != None:
                rmse_probe[b] = test_predict_rating(self, probeset, nbr_samples = 20000)

        return rmse_probe


    def feature_training_online_prototype(self, ratings_index, ratings, initialize_cache = True, initialize_model = True, verbose = False):
        '''
        Compute each features using a Gradient Descent approach. This method is the pure python version of the 
        feature_training() method. It should be only used as a dev tools (very slow). 
        '''
        
        rmse = 2.0
        rmse_last = 2.0

        if initialize_cache:
            self.rating_cache = np.zeros(self.relationship_matrix.get_shape())
        
        if initialize_model:
            self.svd_v = np.zeros([self.dimensionality, self.nbr_users]) + self.feature_init
            self.svd_u = np.zeros([self.dimensionality, self.nbr_items]) + self.feature_init 

        nbr_ratings = ratings.shape[0]
        
        for f in np.arange(self.dimensionality):
            if verbose:
                print "Training the feature " + str(f)
            epoch = 0
            
            while (epoch < self.min_epochs or rmse <= rmse_last - self.min_improvement):
                squared_error = 0.0
                rmse_last = rmse

                for i in xrange(nbr_ratings):
                    user_index = ratings_index[i,0]
                    feature_index = ratings_index[i,1]
                    rating = ratings[i]

                    p = self.estimate_rating(feature_index, user_index, f, self.rating_cache[user_index, feature_index], trailing = 1)
                    error = (1.0 * rating - p);
                    squared_error += error * error;
                    
                    cf = self.svd_v[f,user_index]
                    mf = self.svd_u[f,feature_index]
                    
                    self.svd_v[f,user_index] += self.learning_rate * (error * mf - self.K * cf)
                    self.svd_u[f,feature_index] += self.learning_rate * (error * cf - self.K * mf)
                    
                rmse = np.sqrt(squared_error / nbr_ratings)
                
                if verbose:
                    print "Epoch: " + str(epoch)
                    print "RMSE: " + str(rmse) + "\n"
                
                epoch += 1
                
            for user_index, feature_index, rating in self.ratings_iterator():
                self.rating_cache[user_index, feature_index] = self.estimate_rating(feature_index, user_index, f, self.rating_cache[user_index, feature_index], trailing = 0)
  

    def folding_in_new_user(self, label, ratings):
        '''

        Method for folding-in a new user in the existing model. This method add the new user to the recommendation
        without recalculating the model. The existing ratings of the new user are use for projecting him in the 
        the reduced space. This method is similar to the way add a document/work in Latent Semantic Indexing.
        See the following article for more technical details:

        J. E. Tougas and R. J. Spiteri. Updating the partial singular value decomposition in latent semantic indexing.
        Computational Statistics and Data Analysis, 52(1):174{183, 2007

            * ratings : Existing ratings of the new user
            * label: Label of the user
            
        '''
        new_v = ratings * self.svd_u.T / pow(np.linalg.norm(self.svd_u.T),2)
        return new_v

    
    def folding_in_new_item(self, label, ratings):
        '''

        Method for folding-in a new user in the existing model. This method add the new item to the recommendation
        without recalculating the model. The existing ratings of the new user are use for projecting him in the 
        the reduced space. This method is similar to the way add a document/work in Latent Semantic Indexing.
        See the following article for more technical details:

        J. E. Tougas and R. J. Spiteri. Updating the partial singular value decomposition in latent semantic indexing.
        Computational Statistics and Data Analysis, 52(1):174{183, 2007

            * ratings : Existing ratings of the new item
            * label: Label of the user
            
        '''
        new_u = ratings * self.svd_v.T / pow(np.linalg.norm(self.svd_v.T),2)
        return new_u
            

    def feature_training_folding(self, initialize_model = False, handle_bias = False, verbose = False):
        
        # Initialize the model with previous results if available
        if initialize_model:
            self.svd_v = np.zeros([self.dimensionality, self.nbr_users]) + self.feature_init
            self.svd_u = np.zeros([self.dimensionality, self.nbr_items]) + self.feature_init
        
        ratings_index, ratings = self.get_ratings(randomize_order = True)
        
        if handle_bias:
            
            estimator_loop_with_bias(self.min_epochs, self.max_epochs, self.min_improvement, self.dimensionality, self.feature_init, self.learning_rate,
                                     self.K, self.overall_bias, self.svd_u, self.svd_v, ratings_index, ratings, self.items_bias, self.users_bias, self.nbr_users,
                                     self.nbr_items, int(verbose))
        else:
            estimator_loop_without_bias(self.min_epochs, self.max_epochs, self.min_improvement, self.dimensionality, self.feature_init, self.learning_rate,
                                        self.K, self.svd_u, self.svd_v, ratings_index, ratings, self.nbr_users,
                                        self.nbr_items, int(verbose))
            
        self.compute_components_mean()


    def feature_training_bias(self, initialize_model = True, handle_bias = False, verbose = False):
        '''
        Compute each features using a Gradient Descent approach. This method is the core of the recommender. Once we have
        the matrix containing the rating of the users, we run this method for training the recommendation engine. More precisely,
        this method use a Stochastic Gradient Descent (SGD) for determining the right model parameters. Those model parameters
        are then used by the predictor for computing the ratings for each users-items pair.
        
        The loop intensive part of the code is done in the estimator_loop_without_bias() function, which is a optimize with Cython.
        
            * initialize_model: If True, the model parameters are initialized to zero before the training. If False
              the model parameters already stored are used as the initial value for the training process. [True]
            * handle_bias: Handle the bias if True [False]
            * verbose: Print some info on the terminal if True [False]

        '''
        
        # Initialize the model with previous results if available
        if initialize_model:
            self.svd_v = np.zeros([self.dimensionality, self.nbr_users]) + self.feature_init
            self.svd_u = np.zeros([self.dimensionality, self.nbr_items]) + self.feature_init
        
        ratings_index, ratings = self.get_ratings(randomize_order = True)
        
        #self.items_bias = np.zeros(self.nbr_items) + 0.1
        #self.users_bias = np.zeros(self.nbr_users) + 0.1
        self.compute_overall_avg()
        self.compute_items_bias_bk()
        self.compute_users_bias_bk()

        estimator_loop_with_learned_bias(self.min_epochs, self.max_epochs, self.min_improvement, self.dimensionality, self.feature_init, self.learning_rate,
                                 self.learning_rate_users, self.learning_rate_items, self.K, self.K2, self.overall_bias, self.svd_u, self.svd_v, ratings_index, ratings,
                                 self.items_bias, self.users_bias, self.nbr_users, self.nbr_items, int(verbose))


    def feature_training(self, initialize_model = True, handle_bias = False, verbose = False):
        '''
        Compute each features using a Gradient Descent approach. This method is the core of the recommender. Once we have
        the matrix containing the rating of the users, we run this method for training the recommendation engine. More precisely,
        this method use a Stochastic Gradient Descent (SGD) for determining the right model parameters. Those model parameters
        are then used by the predictor for computing the ratings for each users-items pair.
        
        The loop intensive part of the code is done in the estimator_loop_without_bias() function, which is a optimize with Cython.
        
            * initialize_model: If True, the model parameters are initialized to zero before the training. If False
              the model parameters already stored are used as the initial value for the training process. [True]
            * handle_bias: Handle the bias if True [False]
            * verbose: Print some info on the terminal if True [False]

        '''
        
        # Initialize the model with previous results if available
        if initialize_model:
            self.svd_v = np.zeros([self.dimensionality, self.nbr_users]) + self.feature_init
            self.svd_u = np.zeros([self.dimensionality, self.nbr_items]) + self.feature_init
        
        ratings_index, ratings = self.get_ratings(randomize_order = True)
        
        if handle_bias:
            #self.items_bias = np.zeros(self.nbr_items) + 0.1
            #self.users_bias = np.zeros(self.nbr_users) + 0.1
            self.compute_overall_avg()
            self.compute_items_bias_bk()
            self.compute_users_bias_bk()

            estimator_loop_with_bias(self.min_epochs, self.max_epochs, self.min_improvement, self.dimensionality, self.feature_init, self.learning_rate,
                                     self.learning_rate_users, self.learning_rate_items, self.K, self.overall_bias, self.svd_u, self.svd_v, ratings_index,
                                     ratings, self.items_bias, self.users_bias, self.nbr_users, self.nbr_items, int(verbose))
        else:
            estimator_loop_without_bias(self.min_epochs, self.max_epochs, self.min_improvement, self.dimensionality, self.feature_init, self.learning_rate,
                                        self.K, self.svd_u, self.svd_v, ratings_index, ratings, self.nbr_users,
                                        self.nbr_items, int(verbose))
            

    train = feature_training

    def feature_training_implicit(self, initialize_model = True, verbose = False):
        '''
        Training method handling the Implicit Feedback method
        reference:

        '''

        # Initialize the model with previous results if available
        if initialize_model:
            self.svd_v = np.zeros([self.dimensionality, self.nbr_users]) + self.feature_init
            self.svd_u = np.zeros([self.dimensionality, self.nbr_items]) + self.feature_init
        
        ratings_index, ratings = self.get_ratings(randomize_order = True)
        
        # Precompute the bias
        self.compute_overall_avg()
        self.compute_items_bias_bk()
        self.compute_users_bias_bk() 
        
        # Implicit Feedback
        self.initialize_rated_feedback()
        self.items_feedback = np.zeros([self.dimensionality, self.nbr_items]) 
       
        # Call the estimator with bias and implicit feedback handling
        estimator_loop_with_implicit_feedback(self.min_epochs, self.max_epochs, self.min_improvement, self.dimensionality, self.feature_init, self.learning_rate,
                                              self.learning_rate_users, self.learning_rate_items, self.K, self.overall_bias, self.svd_u, self.svd_v, self.items_feedback,
                                              ratings_index, ratings, self.feedback_rated, self.feedback_hash, 
                                              self.items_bias, self.users_bias, self.nbr_users, self.nbr_items, int(verbose))

    
    def feature_training_dev(self, initialize_model = True, probe = None, verbose = False):
        '''
        Compute each features using a Gradient Descent approach
        '''
        
        # Initialize the model with previous results if available
        rmse = np.zeros(self.max_epochs * self.dimensionality)
        
        if initialize_model:
            self.svd_v = np.zeros([self.dimensionality, self.nbr_users]) + self.feature_init
            self.svd_u = np.zeros([self.dimensionality, self.nbr_items]) + self.feature_init
        
        
        ratings_index, ratings = self.get_ratings(randomize_order = True)
        
        for i, (user_index, feature_index, rating) in enumerate(self.ratings_iterator()):
            ratings_index[i] = [int(user_index), int(feature_index)]
            ratings[i] = rating
       
        estimator_loop(self.min_epochs, self.max_epochs, self.min_improvement, self.dimensionality, self.feature_init, self.learning_rate,
                    self.K, self.svd_u, self.svd_v, ratings_index, ratings, 0, rmse, self.nbr_users, self.nbr_items, int(verbose))

        return rmse

   
    def estimate_rating(self, feature_index, user_index, f, cache = False, trailing = False):
        '''
        Estimate the rating of a known user-item pair (used in the GD algo)
        '''
        if cache and cache > 0:
            sum = cache
        else:
            sum = 1.0
        
        sum += self.svd_u[f,feature_index] * self.svd_v[f,user_index]
        sum = self.clamping(sum)
            
        if trailing:
            sum += (self.dimensionality - f - 1) * self.feature_init * self.feature_init
            sum = self.clamping(sum)
            
        return sum
   

    def predict_rating(self, item_index, user_index):
        '''
        Predict the rating of a unknown user-item pair using the model parameters previously found
        with the feature_training() method. The prediction is simply a dot product with a baseline = 1.0.
            
            * item_index: Internal id of the item
            * user_index : Internal id of the user
        '''
        sum  = np.dot(self.svd_u[:,item_index], self.svd_v[:,user_index]) + 1.0
          
        return sum


    predict = predict_rating
    

    def predict_rating_with_bias(self, item_index, user_index):
        '''
        Predict the rating of a unknown user-item pair using the model parameters previously found
        with the feature_training() method. The prediction is simply a dot product with a BellKor bias baseline.
            
            * item_index: Internal id of the item
            * user_index : Internal id of the user
        '''
        sum  = np.dot(self.svd_u[:,item_index], self.svd_v[:,user_index])
        sum += self.overall_bias + (self.items_bias[item_index] + self.users_bias[user_index])
                
        return sum


    def predict_rating_implicit(self, item_index, user_index):
        '''
        Predict the rating of a unknown user-item pair using the model parameters previously found
        with the feature_training() method. The prediction is simply a dot product with a baseline = 1.0.
            
            * item_index: Internal id of the item
            * user_index : Internal id of the user
        '''
        seek, span = self.feedback_hash[user_index]        
        feedback_norm = 1.0 / np.sqrt(span)
        items_id = self.feedback_rated[seek:seek+span,1]
        
        p_u = self.svd_v[:,user_index] +  feedback_norm * self.items_feedback[:,items_id].sum(axis=1)
        sum = np.dot(self.svd_u[:,item_index], p_u)
        sum += self.overall_bias + (self.items_bias[item_index] + self.users_bias[user_index])

        return sum

    
    def predict_rating_by_label(self, user_label, item_label):
        '''
        Predict the rating of a unknown user-item pair using the model parameters previously found
        with the feature_training() method. The prediction is simply a dot product with a baseline = 1.0.
            
            * feature_index: Internal id of the feature
            * user_index : Internal id of the feature
        '''
        try:
            item_index = self.items_index[item_label]
            user_index = self.users_index[user_label]
            sum  = np.dot(self.svd_u[:,item_index], self.svd_v[:,user_index]) + 1.0
        except KeyError:
            sum = self.baseline_predictor(user_label, item_label)

        return sum


    def _cosine_similarity_binary(self, A_set, B_set):
        count = len(A_set.intersection(B_set))
        return float(count) / np.sqrt(float(len(A_set) * len(B_set)))


    def _compute_users_similarities(self, k):
        self.logger.info('Computing similarities between users')        
        N_csr = self.N.tocsr()
        users_sets_list = []
        for u in range(self.nbr_users):
            row = N_csr.getrow(u)
            nonzero_ids = row.nonzero()[1]
            users_sets_list.append(set(nonzero_ids))

        self.users_similarities_sorted_id = np.zeros((self.nbr_users, k))
        self.users_similarities_values = np.zeros((self.nbr_users, k))
        tes
        for user_index in np.arange(self.nbr_users):
            id, similarities = self.similar_users(user_index, nbr_recommendations = k, similarity_threshold = False, similarities_output = True, method = method)
            if id:
                self.users_similarities_sorted_id[user_index,:] = id
                self.users_similarities_values[user_index,:] = similarities


    def similar_users(self, user_index, nbr_recommendations = 2, similarity_threshold = False, similarities_output = False, method = 'cosine_binary'):
        '''
        Find users similar to an given existing user by looking at the neighbors in the reduced space
        using different method.
        
            * user_index : id of the user
            * nbr_recommendations : Numbers of recommendations in the output
            * similarity_threshold : If specified, the neighbors with similarity < similarity_threshold
              will be excluded
            * similarities_output : If true, the actual value of the similarities will be included 
              in the result
            * method: Method used for computing the similarity ('cosine', 'pearson' or 'euclidean')
            
        note: Dont use cosine similarity with binary matrix
            
        '''
        
        new_user_coord = self.svd_v.T[user_index,0:self.dimensionality]
        similarities = np.array([])
            
        '''
        Assign the method use for computing similarity between vector
        '''
        compute_similarity  = {
          'cosine_binary':     lambda: self._cosine_similarity_binary,                
          'cosine':     lambda: self._cosine_similarity,
          'pearson':    lambda: self._pearson_similarity,
          'euclidean' : lambda : self._euclidean_distance,
        }[method]()
        
        for coord in self.svd_v.T[:,0:self.dimensionality]:
            similarity = compute_similarity(coord, new_user_coord)
            similarities = np.r_[similarities, similarity]
        
        if similarity_threshold:
            pruned_similarities = np.where((similarities > similarity_threshold) == True)[0]
        else:
            pruned_similarities = np.arange(similarities.shape[0])

        similar_item = {}
        for item in pruned_similarities:
            similar_item[item] = similarities[item]
        
        del similar_item[user_index]
        
        sorted_similar_items = sorted(similar_item.iteritems(), key=itemgetter(1), reverse = True)
        nbr_results = len(sorted_similar_items)
        
        if nbr_recommendations == 'All':
            nbr_recommendations = nbr_results
            
        if not similarities_output:
            return [int(i[0]) for i in sorted_similar_items[0:nbr_recommendations]]
        else:
            return [int(i[0]) for i in sorted_similar_items[0:nbr_recommendations]], [i[1] for i in sorted_similar_items[0:nbr_recommendations]]


    def find_user_top_match(self, user_index, nbr_recommendations = 5):
        '''
        Compute all the feature's rating for a given user, sort the result and output the most relevants.

            * user_index: Internal id of the user
            * nbr_recommendations: Numbers of recommendation [5]
        '''
        user_ratings = np.zeros(self.nbr_items)
        self.relationship_matrix_csc = self.relationship_matrix.T.tocsc()

        already_rated = find(self.relationship_matrix_csc[:,user_index])[0]
        already_rated = np.r_[already_rated,user_index]
        
        for i, rating in enumerate(user_ratings):
            if i not in already_rated:
                try:
                    rating = self.predict_rating(i, user_index)
                except Error:
                    rating = 0.0
            else:
                # The rating is not actually zero, we put zero for excluding them from the result
                rating = 0.0
        
            user_ratings[i] = rating
        
        top_results = {}
        nonzero_index = user_ratings.nonzero()[0]
        
        for item in nonzero_index:
            top_results[item] = user_ratings[item]
        
        sorted_top_results = sorted(top_results.iteritems(), key=itemgetter(1), reverse = True)
        
        return [int(i[0]) for i in sorted_top_results[0:nbr_recommendations]], [i[1] for i in sorted_top_results[0:nbr_recommendations]]


    def compute_components_mean(self):
        '''
        '''
        self.components_mean = np.zeros(self.dimensionality)

        for i, c in enumerate(self.svd_u):
            self.components_mean[i] = c.mean()


    def _normalize_cosine_similarity(self, A, B):
        '''
        Compute the similarity between two vector using the cosine similarity method
        '''
        An = A - self.components_mean[1:self.dimensionality]
        Bn = B - self.components_mean[1:self.dimensionality]
        inner_product =  np.inner(An, Bn)
        if inner_product != 0:
            return np.log( 1.0 + (inner_product / (np.linalg.norm(An)*np.linalg.norm(Bn))) )
        else:
            return 0.0


    def similar_items(self, item_index, nbr_recommendations = 2, similarity_threshold = False, similarities_output = False, method = 'pearson'):
        '''
        Find items similar to a given existing item by looking at the neighbors in the reduced space
            * item_index : id of the user
            * nbr_recommendations : Numbers of recommendations in the output
            * similarity_threshold : If specified, the neighbors with similarity < similarity_threshold
              will be excluded
            * similarities_output : If true, the actual value of the similarities will be included 
              in the result
            * method: Method used for computing the similarity ('cosine', 'pearson' or 'euclidean')
            
        '''

        item_coord = self.svd_u[1:self.dimensionality,item_index]
        similarities = np.array([])
        
        '''
        Assign the method use for computing similarity between vector
        '''
        compute_similarity  = {
          'norm_cosine':     lambda: self._normalize_cosine_similarity,
          'cosine':     lambda: self._cosine_similarity,
          'pearson':    lambda: self._pearson_similarity,
          'euclidean' : lambda : self._euclidean_distance,
        }[method]()
        
        for coord in self.svd_u.T:
            similarity = compute_similarity(coord[1:self.dimensionality], item_coord)
            similarities = np.r_[similarities, similarity]
        
        if similarity_threshold:
            pruned_similarities = np.where((similarities > similarity_threshold) == True)[0]
        else:
            pruned_similarities = np.arange(similarities.shape[0])
                        
        similar_item = {}
        for item in pruned_similarities:
            similar_item[item] = similarities[item]
        
        sorted_similar_items = sorted(similar_item.iteritems(), key=itemgetter(1), reverse = True)
        nbr_results = len(sorted_similar_items)
        
        if nbr_recommendations == 'All':
            nbr_recommendations = nbr_results
        
        if not similarities_output:
            return [int(i[0]) for i in sorted_similar_items[1:nbr_recommendations + 1]]
        else:
            return [int(i[0]) for i in sorted_similar_items[1:nbr_recommendations + 1]], [i[1] for i in sorted_similar_items[1:nbr_recommendations + 1]]



    def retrain_user(self, user_index, ratings_index, ratings, verbose = False):
        '''
        Folding-in a new user to the model and retrain the user feature vector
        '''
        # Sanitize the input ratings (keep only the ratings related to the current user)
        valid_ids = np.where(ratings_index[:,0] == user_index)[0]

        self.init_user_features(user_index)

        
        estimator_loop_with_bias_dev(self.min_epochs, self.max_epochs, self.min_improvement, self.dimensionality, self.feature_init, self.learning_rate,
                                 self.learning_rate_users, self.learning_rate_items, self.K, self.overall_bias, self.svd_u, self.svd_v, ratings_index,
                                 ratings[valid_ids,:], self.items_bias, self.users_bias, self.nbr_users, self.nbr_items, 1, 0, int(verbose))


    def retrain_item(self, item_index, ratings_index, ratings, verbose = False):
        '''
        Folding-in a new item to the model and retrain the user feature vector
        '''
        # Sanitize the input ratings (keep only the ratings related to the current item)
        valid_ids = np.where(ratings_index[:,1] == item_index)[0]
        
        self.init_item_features(item_index)

        estimator_loop_with_bias_dev(self.min_epochs, self.max_epochs, self.min_improvement, self.dimensionality, self.feature_init, self.learning_rate,
                                 self.learning_rate_users, self.learning_rate_items, self.K, self.overall_bias, self.svd_u, self.svd_v, ratings_index,
                                 ratings[valid_ids,:], self.items_bias, self.users_bias, self.nbr_users, self.nbr_items, 0, 1, int(verbose))


    def add_user(self, user_label, users_ratings_index, users_ratings):
        '''
        Add a user to the model without adding it into the relationmatrix. The relationmatrix only contains the
        data that is actually use for training the model. The items_ratings add in this method are only used
        for folding-in the user into the existing trained model (a projection into the feature space)
        '''
        if users_ratings_index.shape[0] != users_ratings.shape[0]:
            raise Error('The index and the ratings array must be the same size')
        
        nbr_ratings = users_ratings.shape[0]
        self._nbr_users += 1        
        self.svd_v.resize((self.dimensionality, self._nbr_users), refcheck = False)
        new_id = self._get_new_user_id()
        self.users_index[user_label] = new_id
        self.users_label[new_id] = user_label

        # Construct the ratings array
        ratings_index = np.zeros([nbr_ratings, 2], dtype=np.int32)
        ratings_index[:,0] = new_id
        ratings_index[:,1] = users_ratings_index
    
        # Retrain the user's feature vector
        self.retrain_user(new_id, ratings_index, users_ratings)
        

    def add_item(self, item_label, items_ratings_index, items_ratings):
        '''
        Add an item to the model without adding it into the relationmatrix
        '''
        if items_ratings_index.shape[0] != items_ratings.shape[0]:
            raise Error('The index and the ratings array must be the same size')

        nbr_ratings = items_ratings.shape[0]        
        self._nbr_items += 1        
        self.svd_u.resize((self.dimensionality, self._nbr_items), refcheck = False)
        new_id = self._get_new_item_id()
        self.users_index[item_label] = new_id
        self.users_label[new_id] = item_label

        # Construct the ratings array
        ratings_index = np.zeros([nbr_ratings, 2], dtype=np.int32)
        ratings_index[:,1] = new_id
        ratings_index[:,0] = items_ratings
    
        # Retrain the user's feature vector
        self.retrain_user(new_id, items_ratings_index, items_ratings)

