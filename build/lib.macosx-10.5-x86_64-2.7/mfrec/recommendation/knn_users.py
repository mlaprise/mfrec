'''
.. container:: creation-info

    Created on October 15th, 2010

    @author Martin Laprise

Recommender based on a KNN Users approach

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

class KNNUsersRecommender(BaseRecommender):

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
        BaseRecommender.__init__(self, nbr_users, nbr_items, filename)

        if parameters:
            self.set_parameters(parameters)

        self.nbr_ratings = None
        self.global_avg = None
        self.N = None
        self.items_feedback = None
        self.feedback_rated = None
        self.feedback_hash = None
        self.users_sets_list = None


    def __repr__(self):
        string = 'KNN Users Recommendation Engine\n'
        string += 'Number of users: ' + str(self.nbr_users) + '\n'
        string += 'Number of items: ' + str(self.nbr_items) + '\n'
        return string

    
    def initialize_users_sets(self):
        if self.N == None:
            self.initialize_rated_feedback()
        self.logger.info('Computing similarities between users')        
        N_csr = self.N.tocsr()
        self.users_sets_list = []
        for u in range(self.nbr_users):
            row = N_csr.getrow(u)
            nonzero_ids = row.nonzero()[1]
            self.users_sets_list.append(set(nonzero_ids))
    

    def _cosine_similarity_binary(self, A_set, B_set):
        count = len(A_set.intersection(B_set))
        return float(count) / np.sqrt(float(len(A_set) * len(B_set)))


    def _compute_users_similarities(self, k):
        if self.N == None:
            self.initialize_users_sets()

        self.users_similarities_sorted_id = np.zeros((self.nbr_users, k))
        self.users_similarities_values = np.zeros((self.nbr_users, k))
        
        for user_index in np.arange(self.nbr_users):
            id, similarities = self.similar_users(user_index, nbr_recommendations = k, similarity_threshold = False, similarities_output = True)
            if id:
                self.users_similarities_sorted_id[user_index,:] = id
                self.users_similarities_values[user_index,:] = similarities


    def similar_users(self, user_index, nbr_recommendations = 2, similarity_threshold = False, similarities_output = False):
        '''
        Find users similar to an given existing user by looking at the neighbors in the full space
        using different method.
        
            * user_index : id of the user
            * nbr_recommendations : Numbers of recommendations in the output
            * similarity_threshold : If specified, the neighbors with similarity < similarity_threshold
              will be excluded
            * similarities_output : If true, the actual value of the similarities will be included 
              in the result
            * method: Method used for computing the similarity ('cosine', 'pearson' or 'euclidean')
            
        '''
        
        current_user_set = self.users_sets_list[user_index]
        similarities = np.array([])
            
        for i, user_set in enumerate(self.users_sets_list):
            similarity = self._cosine_similarity_binary(current_user_set, user_set)
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
  

    def find_recommended_items(self, user_index = None, user_label = None, nbr_recommendations = 5, output_label = True):
        if not user_index:
            user_index = self.users_index[user_label]

        k = self.users_similarities_sorted_id.shape[1]
        nearest_neighbors = self.users_similarities_sorted_id[user_index,1:k]
        items_in_neighborhood = find(self.relationship_matrix_csr[nearest_neighbors,:])[1]
        items_count = np.bincount(items_in_neighborhood)
        
        return np.argsort(items_count)[::-1][0:nbr_recommendations], 0


    def train(self, k = 10):
        self._compute_users_similarities(k = k)
        self.relationship_matrix_csr = self.relationship_matrix.tocsr()


    def find_recommended_items_odl(self, user_index = None, user_label = None, nbr_recommendations = 5, output_label = True, predictor = 'predict'):
        if not user_index:
            user_index = self.users_index[user_label]
       
        user_ratings = np.zeros(self.nbr_items)
        self.relationship_matrix_csc = self.relationship_matrix.T.tocsc()

        already_rated = find(self.relationship_matrix_csc[:,user_index])[0]
        already_rated = np.r_[already_rated,user_index]
        
        for i, rating in enumerate(user_ratings):
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


   
    

