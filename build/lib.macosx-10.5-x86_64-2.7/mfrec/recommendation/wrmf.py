'''
.. container:: creation-info

    Created on October 19th, 2010

    @author Martin Laprise

'''
import itertools
from operator import itemgetter

import numpy as np
from scipy.sparse import find

from mfrec.lib.datasets import create_bool_sparse_row, create_bool_sparse_col
from mfrec.recommendation.mf import MFRecommender
from mfrec.lib.als_implicit import als_wrmf


class WRMFRecommender(MFRecommender):
    '''
    Items recommender based on the Weighted matrix factorization method
    Y. Hu, Y. Koren, C. Volinsky: Collaborative filtering for implicit feedback datasets.
	ICDM 2008.
	http://research.yahoo.net/files/HuKorenVolinsky-ICDM08.pdf
    '''

    PARAMETERS_INDEX = {'nbr_epochs' : 'nbr_epochs',
                        'feature_init' : 'feature_init',
                        'regularization_model' : 'K',
                        'neighborhood' : 'neighborhood',
                        'nbr_features' : 'dimensionality'}


    def __init__(self,  nbr_users = 4, nbr_items = 6, parameters = None):
        MFRecommender.__init__(self, nbr_users, nbr_items, parameters)

        # Initialize the training parameters with the default value
        self.nbr_epochs = 20
        self.feature_init = 0.1
        self.K = 0.025
        self.dimensionality = 20
        self.neighborhood = 500

        self.batch_size = 10
        self.rating_cache = None
        self.nbr_ratings = None
        self.global_avg = None
        self.mongodb_iterator = None
        self.components_mean = None
        self.N = None
        self.items_feedback = None
        self.feedback_rated = None
        self.feedback_hash = None

        if parameters:
            self.set_parameters(parameters)


    def __repr__(self):
        string = 'Weighted Regularized Matrix Factorization Recommendation Engine\n'
        string += 'Number of users: ' + str(self.nbr_users) + '\n'
        string += 'Number of items: ' + str(self.nbr_items) + '\n'
        return string


    def predict(self, item_index, user_index):
        sum  = np.dot(self.svd_u[:,item_index], self.svd_v[:,user_index])
        return sum


    def predict_rating_by_label(self, user_label, item_label):
        try:
            item_index = self.items_index[item_label]
            user_index = self.users_index[user_label]
            sum  = np.dot(self.svd_u[:,item_index], self.svd_v[:,user_index])
        except KeyError:
            sum = 0.0

        return sum


    def train(self, initialize_model = True, handle_bias = False, verbose = False):
        '''
        Training method handling the Implicit Feedback method
        '''

        self.relationship_matrix_csc = self.relationship_matrix.T.tocsc()

        # Initialize the model with previous results if available
        if initialize_model:
            self.svd_v = np.zeros([self.dimensionality, self.nbr_users]) + self.feature_init
            self.svd_u = np.zeros([self.dimensionality, self.nbr_items]) + self.feature_init

        m = np.zeros([self.dimensionality, self.dimensionality])
        m_inv = np.zeros([self.dimensionality, self.dimensionality])


        # Implicit Feedback
        self.initialize_rated_feedback()
        self.items_feedback = np.zeros([self.dimensionality, self.nbr_items])
        ratings_users_row, ratings_users_col = create_bool_sparse_row(self.relationship_matrix)
        ratings_items_row, ratings_items_col = create_bool_sparse_col(self.relationship_matrix)

        self.compute_overall_avg()
        # Call the estimator with bias and implicit feedback handling
        als_wrmf(self.nbr_epochs, self.dimensionality, self.svd_u, self.svd_v, m, m_inv,
                 ratings_users_row, ratings_users_col,
                 ratings_items_row, ratings_items_col,
                 self.nbr_users, self.nbr_items, c_pos = 1, k = 0.015, verbose = verbose)
