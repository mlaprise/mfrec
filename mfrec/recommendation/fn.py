'''
.. container:: creation-info

    Created on October 24th, 2010

    @author Martin Laprise

'''
import itertools
from operator import itemgetter

import numpy as np
from scipy.sparse import find

from mfrec.recommendation.base import BaseRecommender
from mfrec.recommendation.mf import MFRecommender


class FNRecommender(MFRecommender):
    '''
    Recommender based on Factorized Neighborhood Model trained with a stochastic gradient descent.

    reference:

    Yehuda Koren
    Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model
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
        BaseRecommender.__init__(self, nbr_users, nbr_items, filename)

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
        string = 'Factorized Neighborhood Recommendation Engine\n'
        string += 'Number of users: ' + str(self.nbr_users) + '\n'
        string += 'Number of items: ' + str(self.nbr_items) + '\n'
        return string


    def predict(self):
        pass

    def train(self, initialize_model = True, verbose = False, kernel = 'train_logistic_kernel'):

        # Initialize the model with random noise from a standard distribution
        if initialize_model:
            self.init_feature_normal(0.0, 0.1)

        ratings = self.get_ratings_dense(randomize_order = True)

        self.compute_overall_avg()
        self.items_bias = np.zeros(self.nbr_items)
        self.users_bias = np.zeros(self.nbr_users)

        eval(kernel)(self.nbr_epochs, self.dimensionality, self.feature_init, self.learning_rate,
                     self.learning_rate_users, self.learning_rate_items, self.K_users, self.K_items, self.K_bias, self.overall_bias,
                     self.svd_u, self.svd_v, ratings, self.items_bias, self.users_bias, 1, 1, int(verbose))




