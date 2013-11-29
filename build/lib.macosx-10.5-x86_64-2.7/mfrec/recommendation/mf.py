'''
.. container:: creation-info

    Created on July 11th, 2010

    @author Martin Laprise

'''
import math
from operator import itemgetter
import itertools
from functools import wraps

import numpy as np
import scipy.sparse
from scipy.sparse import find
from scipy.sparse import lil_matrix, csc_matrix, find

from mfrec.recommendation.base import BaseRecommender
from mfrec.recommendation.metrics import test_predict_rating
from mfrec.lib.gd_estimator import estimator_loop, estimator_loop2, estimator_loop_with_bias, estimator_loop_with_bias_dev, \
                                                           estimator_subloop, predictor_subloop, estimator_loop_without_bias, \
                                                           estimator_loop_with_implicit_feedback


def needs_model(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated_function


class MFRecommender(BaseRecommender):
    '''
    Base class for the matrix factorization based recommender
    '''


    def __init__(self,  nbr_users = 4, nbr_items = 6, parameters = False):
        BaseRecommender.__init__(self, nbr_users, nbr_items, parameters)
        self.neighborhood = 500


    def clamping(self, value, min = 1.0, max = 5.0):
        '''
        Clamp the value between min and max
        '''
        if value > 5.0:
            value = 5.0
        if value < 1.0:
            value = 1.0

        return value


    def train():
        pass


    def predict():
        pass


    def warmyup(self):
        self.relationship_matrix_csc = self.relationship_matrix.tocsc()

    def predict_rating_by_label(self, user_label, item_label, predictor = 'predict_logistic'):
        try:
            item_index = self.items_index[item_label]
            user_index = self.users_index[user_label]
            sum  = getattr(self, predictor)(item_index, user_index)
        except KeyError:
            sum = self.overall_avg

        return sum


    def compute_items_bias_bk(self):
        '''
        Approx. of the regularized items bias as seen in
        BellKor., Advances in Collaborative Filtering
        p. 149 of Recommender Systems Handbook, 2011
        '''
        if not self.overall_bias:
            self.compute_overall_avg()

        self.items_bias = np.zeros(self.nbr_items)
        self.relationship_matrix_csc = self.relationship_matrix.tocsc()

        for i in range(self.nbr_items):
            col = self.relationship_matrix_csc.getcol(i)
            nonzero_ids = col.nonzero()[0]
            NI = len(nonzero_ids)
            if len(nonzero_ids) != 0:
                self.items_bias[i] = (col[nonzero_ids].toarray() - self.overall_bias).sum() / (self.K3 + NI)

        self.items_bias[np.where(np.isnan(self.items_bias))] = 0.0


    def compute_users_bias_bk(self):
        '''
        Approx. of the regularized items bias as seen in
        BellKor, Advances in Collaborative Filtering
        p. 149 of Recommender Systems Handbook, 2011
        '''
        if not self.overall_bias:
            self.compute_overall_avg()
        if self.items_bias == None:
            self.compute_items_bias_bk()

        self.users_bias = np.zeros(self.nbr_users)
        self.relationship_matrix_csr = self.relationship_matrix.tocsr()

        for u in range(self.nbr_users):
            row = self.relationship_matrix_csr.getrow(u)
            nonzero_ids = row.nonzero()[1]
            NI = len(nonzero_ids)
            if len(nonzero_ids) != 0:
                self.users_bias[u] = (row[:,nonzero_ids].toarray() - self.overall_bias - self.items_bias[nonzero_ids]).sum() / (self.K2 + NI)

        self.users_bias[np.where(np.isnan(self.users_bias))] = 0.0


    def init_feature_normal(self, mean = 0.0, std = 0.1):
        '''
        Initialize the features vectors with a random normal distribution

            * mean : Mean of the distribution
            * std : Standard deviation

        '''
        self.svd_u = np.random.normal(mean, std, [self.dimensionality, self.nbr_items])
        self.svd_v = np.random.normal(mean, std, [self.dimensionality, self.nbr_users])


    def init_user_features(self, user_index, mean = 0.0, std = 0.1):
        self.svd_v[:,user_index] = np.random.normal(mean, std, self.dimensionality)


    def init_item_features(self, item_index, mean = 0.0, std = 0.1):
        self.svd_u[:,item_index] = np.random.normal(mean, std, self.dimensionality)


    def find_recommended_items(self, user_index = None, user_label = None, nbr_recommendations = 5, output_label = False, predictor = 'predict'):
        '''
        Compute all the item's rating for a given user, sort the result and output the most relevants.
        Exact same thing as find_user_top_match() but with slightly different input arguments. I keep it seperated
        to avoid breaking the code that uses find_user_to_match().

            * user_index: Internal id of the user
            * nbr_recommendations: Numbers of recommendation [5]
        '''
        if user_index == None:
            user_index = self.users_index[user_label]

        self.neighborhood = min([self.neighborhood, self.nbr_items])

        user_ratings = np.zeros(self.neighborhood)
        test_items = self.get_items_subset(count = self.neighborhood)

        already_rated = find(self.relationship_matrix_csc[:,user_index])[0]
        already_rated = np.r_[already_rated,user_index]

        for i, rating in enumerate(test_items):
            if i not in already_rated:
                try:
                    rating = getattr(self, predictor)(i, user_index)
                except Error:
                    rating = 0.0
            else:
                # The rating is not actually zero, we put zero for excluding them from the result
                rating = 0.0

            user_ratings[i] = rating

        user_ratings[np.where(np.isnan(user_ratings))] = 0.0

        top_results = {}
        nonzero_index = user_ratings.nonzero()[0]

        for item in nonzero_index:
            top_results[item] = user_ratings[item]

        sorted_top_results = sorted(top_results.iteritems(), key=itemgetter(1), reverse = True)

        if output_label:
            items_output = [self.items_label[int(i[0])] for i in sorted_top_results[0:nbr_recommendations]]
        else:
            items_output = [int(i[0]) for i in sorted_top_results[0:nbr_recommendations]]

        items_scores = [i[1] for i in sorted_top_results[0:nbr_recommendations]]

        return items_output, items_scores

