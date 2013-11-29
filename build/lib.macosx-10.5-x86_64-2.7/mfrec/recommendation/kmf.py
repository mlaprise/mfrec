'''
.. container:: creation-info

    Created on October 24th, 2010

    @author Martin Laprise

'''
import itertools
from operator import itemgetter

import numpy as np
from scipy.sparse import find

from mfrec.lib.datasets import create_bool_sparse_row, create_bool_sparse_col
from mfrec.recommendation.base import BaseRecommender
from mfrec.recommendation.mf import MFRecommender
from mfrec.lib.machinelearning.kmf_train import train_logistic_kernel, train_linear_kernel


class KMFRecommender(MFRecommender):
    '''
    Recommender based on Kernel Matrix Factorization trained with a stochastic gradient descent.

    reference:

    Steffen Rendle, Lars Schmidt-Thieme:
	Online-Updating Regularized Kernel Matrix Factorization Models for Large-Scale Recommender Systems.
	RecSys 2008.
	http://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2008-Online_Updating_Regularized_Kernel_Matrix_Factorization_Models.pdf
    '''

    PARAMETERS_INDEX = {'nbr_epochs' : 'nbr_epochs',
                        'min_improvement' : 'min_improvement',
                        'feature_init' : 'feature_init',
                        'learning_rate' : 'learning_rate',
                        'learning_rate_users' : 'learning_rate_users',
                        'learning_rate_items' : 'learning_rate_items',
                        'regularization_users' : 'K',
                        'regularization_items' : 'K2',
                        'regularization_bias' : 'K3',
                        'nbr_features' : 'dimensionality'}


    def __init__(self,  nbr_users = 4, nbr_items = 6, parameters = False, filename = False):
        MFRecommender.__init__(self, nbr_users, nbr_items, filename)

        # Initialize the training parameters with the default value
        self.nbr_epochs = 200
        self.feature_init = 0.1
        self.learning_rate = 0.01
        self.learning_rate_users = 0.01
        self.learning_rate_items = 0.01
        self.K_users = 0.1
        self.K_items = 0.1
        self.K_bias = 0.007
        self.dimensionality = 40

        if parameters:
            self.set_parameters(parameters)

        self.rating_cache = None
        self.nbr_ratings = None
        self.global_avg = None
        self.components_mean = None
        self.N = None
        self.items_feedback = None
        self.feedback_rated = None
        self.feedback_hash = None


    def __repr__(self):
        string = 'Kernel Matrix Factorization Recommendation Engine\n'
        string += 'Number of users: ' + str(self.nbr_users) + '\n'
        string += 'Number of items: ' + str(self.nbr_items) + '\n'
        return string


    def predict_logistic(self, item_index, user_index):
        '''
        Predict method using the logistic kernel
        '''
        sum  = np.dot(self.svd_u[:,item_index], self.svd_v[:,user_index])
        sum += (self.items_bias[item_index] + self.users_bias[user_index])
        return self.min_rating + (1.0 / (1.0 + np.exp(-sum))) * (self.max_rating - self.min_rating)


    def predict_linear(self, item_index, user_index):
        '''
        Predict method using the linear kernel without non-negative constraint
        '''
        sum  = np.dot(self.svd_u[:,item_index], self.svd_v[:,user_index])
        sum += (self.items_bias[item_index] + self.users_bias[user_index])
        return sum


    def predict_linear_neg(self, item_index, user_index):
        '''
        Predict method using the linear kernel without non-negative constraint   
        '''
        sum  = np.dot(self.svd_u[:,item_index], self.svd_v[:,user_index])
        sum += (self.items_bias[item_index] + self.users_bias[user_index])
        return self.min_rating  + sum * (self.max_rating - self.min_rating)


    predict = predict_logistic


    def predict_rating_by_label(self, user_label, item_label, predictor = 'predict_logistic'):
        try:
            item_index = self.items_index[item_label]
            user_index = self.users_index[user_label]
            sum  = getattr(self, predictor)(item_index, user_index)
        except KeyError:
            sum = self.overall_avg

        return sum


    def retrain_user(self, user_index, ratings_index, ratings, verbose = False, kernel = 'train_logistic_kernel'):
        '''
        Folding-in a new user to the model and retrain the user feature vector
        '''
        # Sanitize the input ratings (keep only the ratings related to the current user)
        valid_ids = np.where(ratings_index[:,0] == user_index)[0]

        self.init_user_features(user_index)

        eval(kernel)(self.nbr_epochs, self.dimensionality, self.feature_init, self.learning_rate,
                     self.learning_rate_users, self.learning_rate_items, self.K_users, self.K_items, self.K_bias, self.overall_bias,
                     self.svd_u, self.svd_v, ratings_index[valid_ids,:], ratings[valid_ids], self.items_bias, self.users_bias, 1, 0, int(verbose))



    def retrain_item(self, item_index, ratings_index, ratings, verbose = False, kernel = 'train_logistic_kernel'):
        '''
        Folding-in a new item to the model and retrain the user feature vector
        '''
        # Sanitize the input ratings (keep only the ratings related to the current item)
        valid_ids = np.where(ratings_index[:,1] == item_index)[0]
        
        self.init_item_features(item_index)

        eval(kernel)(self.nbr_epochs, self.dimensionality, self.feature_init, self.learning_rate,
                     self.learning_rate_users, self.learning_rate_items, self.K_users, self.K_items, self.K_bias, self.overall_bias,
                     self.svd_u, self.svd_v, ratings[valid_ids,:], self.items_bias, self.users_bias, 0, 1, int(verbose))


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


    def train(self, initialize_model = True, verbose = False, kernel = 'train_logistic_kernel'):
        '''
        Train the kernel matrix factorization model using a stochastic gradient descent

            * initialize_model : Initialize the feature vector if == True [True]
            * verbose : Print info if == True [False]
            * kernel : Kernel used for the training: 'train_linear_kernel', 'train_logistic_kernel' ['train_logistic_kernel']
        '''

        # Initialize the model with random noise from a standard distribution
        self.relationship_matrix_csc = self.relationship_matrix.T.tocsc()
        
        if initialize_model:
            self.init_feature_normal(0.0, 0.1) 

        ratings_index, ratings = self.get_ratings(randomize_order = True)
        
        self.compute_overall_avg()
        self.items_bias = np.zeros(self.nbr_items)
        self.users_bias = np.zeros(self.nbr_users)
 
        eval(kernel)(self.nbr_epochs, self.dimensionality, self.feature_init, self.learning_rate,
                     self.learning_rate_users, self.learning_rate_items, self.K_users, self.K_items, self.K_bias, self.overall_bias,
                     self.svd_u, self.svd_v, ratings_index, ratings, self.items_bias, self.users_bias, 1, 1, int(verbose))
