'''
.. container:: creation-info

    Created on October 15th, 2010

    @author Martin Laprise

'''
import itertools

import numpy as np
from scipy.sparse import find

from mfrec.recommendation.base import BaseRecommender

class RandomRecommender(BaseRecommender):
    '''
    Simple 'recommender' base on most popular items. For testing metrics only !
    '''

    PARAMETERS_INDEX = {}

    def __init__(self,  nbr_users = 4, nbr_items = 6, parameters = False, filename = False):
        BaseRecommender.__init__(self, nbr_users, nbr_items, filename)

        if parameters:
            self.set_parameters(parameters)
        
        self.relationship_matrix_dok = None


    def __repr__(self):
        string = 'Random Recommendation Engine\n'
        string += 'Number of users: ' + str(self.nbr_users) + '\n'
        string += 'Number of items: ' + str(self.nbr_items) + '\n'
        return string


    def set_parameters(self, parameters):
        '''
        Set the parameters for the training
        '''
        for k, v in parameters.iteritems():
            try:
                setattr(self, self.PARAMETERS_INDEX[k], v)
            except KeyError:
                raise Error('Wrong parameters')
    

    def train(self):
        self.relationship_matrix_dok = self.relationship_matrix.todok()


    def find_recommended_items(self, user_index, nbr_recommendations = 10, output_label = False):
        recommended_items = []
        i = 0 
        while (len(recommended_items) < nbr_recommendations):
            recommended_item = np.random.randint(1,self.nbr_items)
            if not self.relationship_matrix_dok.has_key((user_index, recommended_item)) or recommended_item not in recommended_items:
                recommended_items.append(recommended_item)
            i += 1
        return recommended_items, 0


