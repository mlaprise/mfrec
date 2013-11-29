'''
    Recommender based on a Singular Value Decomposition

    Created on July 11th, 2010

    @author: Martin Laprise

'''
from operator import itemgetter

import numpy as np
from scipy.sparse import find
from sparsesvd import sparsesvd

from mfrec.recommendation.mf import MFRecommender


class SVDRecommender(MFRecommender):
    '''

    Simple Recommender base on a Singular Value Decomposition

    A Singular Value Decomposition is use for reducing the dimensionality of the space.
    The optimal dimensionality depends on the nature of the problem (aka the dataset). The optimal k value
    depends on the nature of the problem and on the sparsity on the relation matrix.

    Since this is pure singular value decomposition (not a regularized matrix factorisation) the item feature vector
    is not self.svd_u but self.svd_u * self.svd_s

    The class include some knn method for a hybrid svd-knn recommender
    '''

    PARAMETERS_INDEX = {'nbr_features' : 'dimensionality'}


    def __init__(self,  nbr_users = 4, nbr_items = 6, parameters = False, filename = False):
        MFRecommender.__init__(self, nbr_users, nbr_items, filename)

        # Initialize the training parameters with the default value
        self.dimensionality = 150

        if parameters:
            self.set_parameters(parameters)


    def __repr__(self):
        string = 'Simple SVD Recommendation Engine\n'
        string += 'Number of users: ' + str(self.nbr_users) + '\n'
        string += 'Number of items: ' + str(self.nbr_items) + '\n'
        string += 'Dimensionality: ' + str(self.dimensionality) + '\n'
        return string


    def train(self):
        self._compute_svd(normalize_data = True)


    def predict(self, item_index, user_index):
        ru = self.relationship_matrix.getrow(user_index).toarray()
        qi_t = self.svd_v[:,item_index]
        Q = self.svd_v.T
        a = np.dot(ru, Q)
        b = np.dot(a, qi_t)
        return b


    @property
    def svd_full_s(self):
        '''
        Recover the diagonalized S matrix
        '''
        return np.diag(self.svd_s)


    def _compute_svd(self, normalize_data = True):
        self.logger.info('Computing the Singular Value Decomposition of the relation matrix')

        if normalize_data:
            self.data_normalization()

        self.relationship_matrix_csc = self.relationship_matrix.tocsc()
        self.svd_u, self.svd_s, self.svd_v = sparsesvd(self.relationship_matrix_csc, self.dimensionality)


    def _compute_items_similarities(self, k):
        '''
        Precomputes similarities between items
        '''
        self.logger.info('Computing similarities between items')
        self.items_similarities_sorted_id = np.zeros((self.nbr_items, k))
        self.items_similarities_values = np.zeros((self.nbr_items, k))

        for item_index in np.arange(self.nbr_items):
            id, similarities = self.similar_items(item_index, k, False, 0.0, True, 'cosine')
            if id:
                self.items_similarities_sorted_id[item_index,:] = id
                self.items_similarities_values[item_index,:] = similarities


    def _compute_users_similarities(self, k, method = 'cosine'):
        '''
        Precomputes similarities between users
        @todo: Save only rated
        '''
        self.logger.info('Computing similarities between users')
        self.users_similarities_sorted_id = np.zeros((self.nbr_users, k))
        self.users_similarities_values = np.zeros((self.nbr_users, k))

        for user_index in np.arange(self.nbr_users):
            id, similarities = self.similar_users(user_index, nbr_recommendations = k, similarity_threshold = False, similarities_output = True, method = method)
            if id:
                self.users_similarities_sorted_id[user_index,:] = id
                self.users_similarities_values[user_index,:] = similarities


