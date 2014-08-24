'''
    Basic Recommendation Engine

    Created on May 25th, 2010

    @author: Martin Laprise

'''
import logging
import pickle
import itertools
import datetime
from operator import itemgetter

import numpy as np
from pymongo import Connection
import scipy.io
import scipy.stats
from scipy.sparse import csc_matrix, find
from sparsesvd import sparsesvd
from scipy.sparse import lil_matrix

class Error(Exception): pass

class LinearModel(object):
    '''
    Base class for all the linear prediction model
    '''
    _logger_name = 'mfrec.linearmodel'

    def __init__(self):
        object.__init__(self)
        self.logger = logging.getLogger(self._logger_name)
        # Sparse Matrix containing the training data
        self.relationship_matrix = None
        self.relationship_matrix_csc = None
        self.relationship_matrix_csr = None

        # u and v are weight parameters of the linear model
        self.svd_s = None
        self.svd_u = None
        self.svd_v = None

        # The number of feature of the model
        self.dimensionality = 40


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
        pass


    def predict(self):
        pass



class BaseRecommender(object):
    '''

    Recommendation Engine

    Encapsulates most of the mechanics related to the recommendation of a
    a item to a user. The central part of the class is the relation matrix which stores the relation
    of each user to each item via a rating value. Because this matrix is a highly sparsed matrix
    the rating are stored in a Scipy Sparse Matrix for memory efficiency.

    '''

    PARAMETERS_INDEX = {}

    _logger_name = 'mfrec.recommender'

    def __init__(self, nbr_users = 4, nbr_items = 6, parameters = None):
        object.__init__(self)
        self.logger = logging.getLogger(self._logger_name)
        self.logger.info('Initializing a Recommendation Engine')

        # The number of latent features to use in an SVD-based recommender.
        self.dimensionality = 40

        self.min_rating = 1.0
        self.max_rating = 5.0
        self.items_similarities_sorted_id = None
        self.items_similarities_values = None
        self.users_similarities_sorted_id = None
        self.users_similarities_values = None
        self.relationship_matrix = None
        self.relationship_matrix_csc = None
        self.relationship_matrix_csr = None
        self.items_index = {}
        self.items_label = []
        self.items_index_map = {}
        self.users_index = {}
        self.users_label = []
        self.svd_u = None
        self.svd_s = None
        self.svd_v = None
        self._svd_full_s = None
        self._nbr_users = None
        self._nbr_items = None
        self.warmedup = False
        self.data_normalized = False
        self.users_bias = None
        self.items_bias = None
        self.overall_bias = None
        self.items_avg = None
        self.user_top_match = None
        self.db_batch_size = 1000
        self.sorted_items_by_count = None
        self.N = None
        self.items_feedback = None
        self.feedback_rated = None
        self.feedback_hash = None
        self.metadata = {}

        if parameters:
            self.set_parameters(parameters)

        self.initialize_relationship_matrix(int(nbr_users), int(nbr_items))

        self.db_connection = None
        self.db = None


    def __repr__(self):
        string = 'Generic Recommendation Engine\n'
        string += 'Number of users: ' + str(self.nbr_users) + '\n'
        string += 'Number of items: ' + str(self.nbr_items) + '\n'
        string += 'Dimensionality: ' + str(self.dimensionality) + '\n'
        return string


    def initialize_model(self):
        self.svd_u = np.zeros([self.dimensionality, self.nbr_items])
        self.svd_v = np.zeros([self.dimensionality, self.nbr_users])


    def initialize_bias(self):
        self.items_bias = np.zeros([self.nbr_items])
        self.users_bias = np.zeros([self.nbr_users])


    def _find_item_label(self, index):
        """return the key of dictionary dic given the value"""
        dic = self.items_index
        if isinstance(index, list):
            result = []
            for i in index:
                result.append([k for k, v in dic.iteritems() if v == i][0])
            return result
        else:
            return [k for k, v in dic.iteritems() if v == index][0]

    def set_name(self, name):
        self.metadata['model_name'] = name

    @property
    def nbr_users(self):
        self._nbr_users = len(self.users_label)
        return self._nbr_users


    @property
    def nbr_items(self):
        self._nbr_items = len(self.items_label)
        return self._nbr_items


    def set_parameters(self, parameters):
        '''

        Set the parameters for the training

            * parameters : Dictionary containing the parameters for the training
                * min_epochs : Minimal number of epochs performs in the Gradient Descent process
                * max_epochs : Maximal number of epochs performs in the Gradient Descent process
                * min_improvement: Minimal improvement for each epoch, if the improvement of the RMSE on the training set in inferior to this
                  value, the process is stoped and we go to the next feature
                * learning_rate: Learning Rate of the Gradient Descent
                * regularization: Regularization parameter use to avoid overfitting over the training set
                * nbr_features: Number of features

        '''
        for k, v in parameters.iteritems():
            try:
                setattr(self, self.PARAMETERS_INDEX[k], v)
            except KeyError:
                raise Error('Wrong parameters')


    def get_nbr_ratings(self):
        '''
        Return the number of non-zero item in the relationshion_matrix (nbr. of ratings).
        '''
        return find(self.relationship_matrix)[1].shape[0]


    def initialize_from_file(self, filename):
        '''
        Initialize the main relationship matrix and assigns default labels to each items
        '''

        self.logger.info('Initializing the recommendation engine from a saved state')

        self.load_state(filename)

        nbr_users, nbr_items = self.relationship_matrix.shape

        for item in xrange(nbr_items):
            self.items_label.append('item' + str(item))
            self.items_index['item' + str(int(item))] = int(item)

        for user in range(nbr_users):
            self.users_label.append('user' + str(user))
            self.users_index['user' + str(user)] = user

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


    def initialize_rated_feedback(self):
        self.N = self.relationship_matrix.astype(bool)
        self.feedback_rated, self.feedback_hash = self.get_feedback()


    def initialize_relationship_matrix(self, nbr_users, nbr_items):
        '''
        Initialize the main relationship matrix and assigns default labels to each item
        '''

        self.logger.info('Initializing the relationship matrix')

        self.relationship_matrix = lil_matrix((nbr_users, nbr_items))

        for item in xrange(nbr_items):
            self.items_label.append('item' + str(item))
            self.items_index['item' + str(int(item))] = int(item)

        for user in range(nbr_users):
            self.users_label.append('user' + str(user))
            self.users_index['user' + str(user)] = user


    def ratings_iterator(self):
        cx = self.relationship_matrix.tocoo()
        return itertools.izip(cx.row, cx.col, cx.data)


    def set_dimensionality(self, new_dim_value):
        self.dimensionality = new_dim_value


    def change_dimensionality(self, new_dim_value):
        self.dimensionality = new_dim_value
        self._compute_svd(normalize_data = False)


    def _mean_centering(self, input_ratings):
        '''
        Data normalization using mean centering
        '''
        non_zero_ids = input_ratings.nonzero()[0]
        mean_ratings = input_ratings[non_zero_ids].mean()
        return input_ratings - mean_ratings, mean_ratings


    def _range_scaling(self, input_ratings, input_range = False, output_range = [1.0, 5.0]):
        '''
        Scaling of the date in a different range
        '''
        r_min = output_range[0]
        r_max = output_range[1]

        if not input_range:
            d_max = input_ratings.max()
            d_min = input_ratings.min()
        else:
            d_min = input_range[0]
            d_max = input_range[1]

        return input_ratings * ((r_max - r_min) / (d_max - d_min)) + ((r_min*d_max - r_max*d_min) / (d_max - d_min))


    def data_normalization(self, users_based = True, items_based = False):
        '''
        Inplace data normalization using mean centering on users ratings for alleviated
        the effect the user bias

        @todo : Remove items bias (same thing but on column ... much slower performance
                because of the sparsed matrix format)
        '''
        self.users_bias = np.zeros(self.nbr_users)
        self.logger.info('Normalizing data using mean centering method on users ratings')
        for i, user_ratings in enumerate(self.relationship_matrix):
            non_zero_ids = user_ratings.nonzero()[1]
            if len(non_zero_ids) != 0:
                mean_centered, self.users_bias[i] = self._mean_centering(self.relationship_matrix[i,non_zero_ids].todense())
                self.relationship_matrix[i,non_zero_ids] = mean_centered
        '''
        self.items_bias = np.zeros(self.nbr_items)
        self.logger.info('Normalizing data using mean centering method on items ratings')
        self.relationship_matrix_csc = self.relationship_matrix.tocsc()
        for i, item_ratings in enumerate(self.relationship_matrix_csc):
            non_zero_ids = item_ratings.nonzero()[1]
            if len(non_zero_ids) != 0:
                mean_centered, self.items_bias[i] = self._mean_centering(self.relationship_matrix[i,non_zero_ids].todense())
                self.relationship_matrix[i,non_zero_ids] = mean_centered
        '''

        self.data_normalized = True


    def find_rating_scale(self):
        '''
        Find the new rating scale for a user after a user-bias normalisation
        '''
        pass


    def prune_rating_post_training(self, nbr_min_rating = 20):
         self.logger.info('Recommendation : Removing the user/item with less then' + str(nbr_min_rating) + 'ratings')
         for i, v in enumerate(self.relationship_matrix.T):
            if v.nonzero()[0].shape[0] < nbr_min_rating:
                self.svd_u[:,i] = np.nan

         for i, v in enumerate(self.relationship_matrix):
            if v.nonzero()[0].shape[0] < nbr_min_rating:
                self.svd_v[:,i] = np.nan


    def prune_rating_pre_training(self, nbr_min_rating = 20):
         self.logger.info('Training : Removing the user/item with less then' + str(nbr_min_rating) + 'ratings')
         for i, v in enumerate(self.relationship_matrix.T):
            if v.nonzero()[0].shape[0] < nbr_min_rating:
                self.relationship_matrix[:,i] = 0

         for i, v in enumerate(self.relationship_matrix):
            if v.nonzero()[0].shape[0] < nbr_min_rating:
                self.relationship_matrix[i,:] = 0



    def compute_means(self):
        '''
        OBSELETE ??
        Compute user mean and item mean
        '''

        self.users_bias = np.zeros(self.nbr_users + 1)
        self.items_bias = np.zeros(self.nbr_users + 1)
        self.logger.info('Normalizing data using mean centering method on users ratings')

        for i, user_ratings in enumerate(self.relationship_matrix):
            non_zero_ids = user_ratings.nonzero()[1]
            if len(non_zero_ids) != 0:
                self.users_bias[i+1] = self.relationship_matrix[i,non_zero_ids].mean()

        for i, item_ratings in enumerate(self.relationship_matrix.T):
            non_zero_ids = item_ratings.nonzero()[1]
            if len(non_zero_ids) != 0:
                self.items_bias[i+1] = self.relationship_matrix[i,non_zero_ids].mean()


    def compute_item_pseudo_avg(self):
        '''
        Compute a pseudo average for as a baseline predictor
        '''
        self.overall_avg = find(self.relationship_matrix)[2].mean()
        self.items_avg = np.zeros(self.nbr_items)
        variance_ratio = 25.0
        for i, item in enumerate(self.relationship_matrix.T):
            ids = item.nonzero()[1]
            nbr_ratings = len(ids)
            try:
                self.items_avg[i] = (self.overall_avg * variance_ratio + item[:,ids].sum()) / (variance_ratio + nbr_ratings)
            except ValueError:
                pass


    def users_average(self, user_label):
        '''
        Compute the users average
        '''
        user_index = self.users_index[user_label]
        average = find(self.relationship_matrix.getrow(user_index))[2].mean()
        return average


    def items_average(self, item_label):
        '''
        Compute the items average
        '''
        try:
            item_index = self.items_index[item_label]
            average = find(self.relationship_matrix_csc.getcol(item_index))[2].mean()
        except:
            self.relationship_matrix_csc = self.relationship_matrix.tocsc()
            item_index = self.items_index[item_label]
            average = find(self.relationship_matrix_csc.getcol(item_index))[2].mean()

        return average


    def baseline_predictor(self, user_label, item_label):
        '''
        Predict the rating of a unknown user-item pair without any training. If the recommender can't
        performs a ratings prediction we can fall back on the baseline predictor. The baseline predictor
        simply use the items mean as a prediction and fall back on the user mean if the items is not in the
        model yet.

        Baseline predictors =>  do not involve user-item interaction.
        '''
        try:
            baseline_prediction = self.items_average(item_label)
        except KeyError:
            baseline_prediction = self.users_average(user_label)

        return baseline_prediction


    def baseline_predictor2(self, item_index, user_index):
        '''
        Baseline predictor using the overall avg + item deviation from the average + user deviation from the average
        '''
        baseline_predictor = self.overall_bias + self.items_bias[item_index] + self.users_bias[user_index]
        return baseline_predictor


    def baseline_predictor3(self, item_index, user_index):
        '''
        Baseline using random number ... for dev purpose obviously
        '''
        return np.random.randint(1,5)


    def baseline_predictor4(self, item_index, user_index):
        '''
        Baseline predictor using the item avg + user deviation from the average
        '''
        baseline_predictor = self.items_avg[item_index] + self.users_bias[user_index]
        return baseline_predictor


    def compute_items_bias(self):
        '''
        Compute the average ratings of a given item.
        '''
        if not self.overall_bias:
            self.compute_overall_avg()

        self.relationship_matrix_csc = self.relationship_matrix.tocsc()
        self.items_bias = np.zeros(self.nbr_items)
        self.items_avg = np.zeros(self.nbr_items)

        for i in range(self.nbr_items):
            self.items_avg[i] = find(self.relationship_matrix_csc[:,i])[2].astype(float).mean()
            #self.items_bias[i] = find(self.relationship_matrix_csc[:,i])[2].astype(float).mean() - self.overall_bias
            self.items_bias[i] = self.items_avg[i] - self.overall_bias

        self.items_bias[np.where(np.isnan(self.items_bias))] = 0.0
        self.items_avg[np.where(np.isnan(self.items_avg))] = 0.0


    def compute_overall_avg(self):
        '''
        Compute the average ratings
        '''
        self.overall_bias = find(self.relationship_matrix)[2].astype(float).mean()


    def compute_users_bias(self):
        '''
        Compute the average ratings of a given user.
        '''
        if not self.overall_bias:
            self.compute_overall_avg()

        self.users_bias = np.zeros(self.nbr_users)
        for u in range(self.nbr_users):
            self.users_bias[u] = find(self.relationship_matrix[u,:])[2].astype(float).mean() - self.overall_bias

        self.users_bias[np.where(np.isnan(self.users_bias))] = 0.0


    def compute_items_avg(self):
        '''
        Compute the average ratings of a given item.
        '''
        if not self.overall_bias:
            self.compute_overall_avg()

        self.relationship_matrix_csc = self.relationship_matrix.tocsc()
        self.items_bias = np.zeros(self.nbr_items)
        for i in range(self.nbr_items):
            self.items_avg[i] = find(self.relationship_matrix_csc[:,i])[2].astype(float).mean()

        self.items_bias[np.where(np.isnan(self.items_bias))] = 0.0


    def data_normalization_item(self):
        '''
        Inplace data normalization using mean centering on items ratings for alleviated
        the effect the item bias. The item bias is generally less important then user bias
        '''
        self.compute_item_mean()
        self.logger.info('Normalizing data using mean centering method on items ratings')
        for i in range(self.nbr_items):
            non_zero_ids = self.relationship_matrix[:,i].nonzero()[0]
            if len(non_zero_ids) != 0:
                non_norm_value = self.relationship_matrix[non_zero_ids,i].toarray()
                norm_value = non_norm_value - self.items_bias[i]
                self.relationship_matrix[non_zero_ids, i] = norm_value

        self.data_normalized = True


    def save_state(self, filename):
        '''
        Save the state of the recommendation engine in files
        '''
        scipy.io.mmwrite(filename + '_relations.mtx', self.relationship_matrix)
        np.savez(filename + '_svd.npz', svd_s = self.svd_s, svd_u = self.svd_u, svd_v = self.svd_v)
        self.logger.info('The state of the recommendation engine has been saved.')

        f = file(filename + '_userslabel.dat', 'w')
        pickle.dump(self.users_label, f)
        f = file(filename + '_usersindex.dat', 'w')
        pickle.dump(self.users_index, f)
        f = file(filename + '_itemslabel.dat', 'w')
        pickle.dump(self.items_label, f)
        f = file(filename + '_itemsindex.dat', 'w')
        pickle.dump(self.items_index, f)
        f = file(filename + '_itemsindexmap.dat', 'w')
        pickle.dump(self.items_index_map, f)


    def load_state(self, filename):
        '''
        Load a previous state of the recommendation engine from files
        '''
        svd = np.load(filename + '_svd.npz')
        self.svd_u = svd['svd_u']
        self.svd_s = svd['svd_s']
        self.svd_v = svd['svd_v']
        self.relationship_matrix = scipy.io.mmread(filename + '_relations.mtx')
        f = open(filename + '_usersindex.dat')
        self.users_index = pickle.load(f)
        f = open(filename + '_userslabel.dat')
        self.users_label = pickle.load(f)
        f = open(filename + '_itemsindex.dat')
        self.items_index = pickle.load(f)
        f = open(filename + '_itemslabel.dat')
        self.items_label = pickle.load(f)
        f = open(filename + '_itemsindexmap.dat')
        self.items_index_map = pickle.load(f)
        self.logger.info('The state of the recommendation engine has been loaded')


    def save_items_to_db(self):
        '''
        Save the items model into the mongodb

        Schema:
            lbl : Label (string)
        '''
        item_batch = []

        for item_label in self.items_label:
            item  = {}
            item['lbl'] = item_label
            item_index = self.items_index[item_label]
            item['w'] = self.svd_u[:,item_index].tolist()
            item['lst_up'] = datetime.datetime.utcnow()
            item_batch.append(item)

        self.db.items.insert(item_batch)


    def save_ratings_graph_to_neo4j(self):
        '''
        Stored the items in a graph database
        '''
        from neo4jrestclient.client import GraphDatabase

        gdb = GraphDatabase("http://localhost:7474/db/data/")
        user_db_index = {}
        item_db_index = {}

        for user_label in self.users_label:
            user_index = self.users_index[user_label]
            node = gdb.nodes.create(type = 'User', label = user_label)
            user_db_index[user_index] = node.id

        for item_label in self.items_label:
            item_index = self.items_index[item_label]
            node = gdb.nodes.create(type = 'Item', label = item_label)
            item_db_index[item_index] = node.id

        ratings_itr = self.ratings_iterator()

        for user_index, item_index, rating_value in ratings_itr:
            user_db_id = user_db_index[user_index]
            item_db_id = item_db_index[item_index]
            relation = gdb.node[user_db_id].relationships.create('rating', gdb.node[item_db_id], value=rating_value)


    def save_ratings_graph_to_emb_neo4j(self):
        '''
        Stored the items in a graph database
        '''

        from neo4j import GraphDatabase

        gdb = GraphDatabase("/Users/mlaprise/graph_db")
        user_db_index = {}
        item_db_index = {}

        for user_label in self.users_label:
            user_index = self.users_index[user_label]
            with gdb.transaction:
                node = gdb.node(type = 'User', label = user_label)

            user_db_index[user_index] = node.id

        for item_label in self.items_label:
            item_index = self.items_index[item_label]
            with gdb.transaction:
                node = gdb.node(type = 'Item', label = item_label)
            item_db_index[item_index] = node.id

        ratings_itr = self.ratings_iterator()

        for user_index, item_index, rating_value in ratings_itr:
            user_db_id = user_db_index[user_index]
            item_db_id = item_db_index[item_index]
            with gdb.transaction:
                user_node = gdb.node[user_db_id]
                item_node = gdb.node[item_db_id]
                relationship = user_node.rating(item_node, value = rating_value)

        gdb.shutdown()

    def save_users_to_db(self):
        '''
        Save the user model into the mongodb

        Schema:
            lbl : Label (string)

        '''
        user_batch = []

        for user_label in self.users_label:
            user  = {}
            user['lbl'] = user_label
            user_index = self.users_index[user_label]
            user['w'] = self.svd_v[:,user_index].tolist()
            user['lst_up'] = datetime.datetime.utcnow()
            user_batch.append(user)

        self.db.users.insert(user_batch)


    def update_model_in_db(self):
        self.db.models.update({'name': self.metadata['model_name']}, {'$set': {'bias' : self.overall_bias}})


    def update_users_model_in_db(self):
        '''
        Save the model parameters into the mongodb
        '''

        for user_label in self.users_label:
            user_index = self.users_index[user_label]
            # Store the model parameters in the db
            w = self.svd_v[:,user_index].tolist()
            bu = self.users_bias[user_index]
            self.db.users.update({'lbl':user_label, 'mod':self.metadata['model_name'] }, {'$set':{'w': w, 'bu': bu}})


    def update_items_model_in_db(self):
        '''
        Save the model parameters into the mongodb
        '''

        for item_label in self.items_label:
            item_index = self.items_index[item_label]
            w = self.svd_u[:,item_index].tolist()
            bi = self.items_bias[item_index]
            self.db.items.update({'lbl':item_label, 'mod':self.metadata['model_name'] }, {'$set': {'w': w, 'bi': bi}})


    def save_ratings_to_db(self):
        ratings_itr = self.ratings_iterator()
        nbr_ratings = self.get_nbr_ratings()
        nbr_batchs = nbr_ratings / self.db_batch_size

        for i in xrange(nbr_batchs):
            rating_batch = []
            for user_index, item_index, rating_value in ratings_itr:
                rating = {}
                rating['u_lbl'] = self.users_label[user_index]
                rating['i_lbl'] = self.items_label[item_index]
                rating['val'] = rating_value
                rating['lst_up'] = datetime.datetime.utcnow()
                rating_batch.append(rating)
            if rating_batch:
                self.db.ratings.insert(rating_batch)


    def set_users_from_db(self, mongo_itr):
        '''
        Set user using a mongodb iterator (cursor)
        '''
        self.clear_users_index()
        for i, user in enumerate(mongo_itr):
            self.svd_v[:,i] = user['w']
            self.users_bias[i] = user['bu']
            self.users_label[i] = user['lbl']
            self.users_index[user['lbl']] = i
            self.logger.debug(user['lbl'] + 'added')


    def set_items_from_db(self, mongo_itr):
        '''
        Set users using a mongodb iterator (cursor)
        '''
        self.clear_items_index()
        for i, item in enumerate(mongo_itr):
            self.svd_u[:,i] = item['w']
            self.items_bias[i] = item['bi']
            self.items_label[i] = item['lbl']
            self.items_index[item['lbl']] = i
            self.logger.debug(item['lbl'] + 'added')


    def set_ratings_from_db(self, mongo_itr):
        '''
        Set the items
        '''
        for item in mongo_itr:
            try:
                item_index = self.items_index[item['i_lbl']]
                user_index = self.users_index[item['u_lbl']]
            except KeyError:
                self.logger.debug('No correspond item or user')
                pass
            try:
                self.relationship_matrix[user_index, item_index] = item['val']
                self.logger.debug('Rating added: ' + str(user_index) + ',' + str(item_index) + ' :' + str(item['val']))
            except:
                print 'Error setting rating'

        self.relationship_matrix_csc = self.relationship_matrix.tocsc() 


    def clear_users_index(self):
        self.users_index = {}


    def clear_items_index(self):
        self.items_index = {}


    def save_model_snapshot(self, filename):
        np.savez(filename + '_model_snapshot.npz', svd_u = self.svd_u, svd_v = self.svd_v)


    def load_model_snapshot(self, filename):
        svd = np.load(filename + '_model_snapshot.npz')
        self.svd_u = svd['svd_u']
        self.svd_v = svd['svd_v']


    def set_item(self, user, items_list):
        '''
        Assign values to a set of item for a specific user
        '''
        for item in items_list:
            self.relationship_matrix[int(self.users_index[user]), int(self.items_index[item['label']])]   = float(item['value'])


    def set_item_by_id(self, user_index, item_index, value):
        '''
        Assign a single value to a item for a specific user
        '''
        self.relationship_matrix[user_index, item_index] = float(value)
        self.logger.debug('Set item: user ' + str(user_index) + ' , ' + 'item ' + str(item_index) + ' -> ' + str(value))


    def set_item_by_label(self, user, item, value):
        '''
        Assign a single value to a item for a specific user
        '''
        self.relationship_matrix[int(self.users_index[user]), int(self.items_index[item])] = float(value)
        self.logger.debug('Set item: user ' + user + ' , ' + 'item ' + item + ' -> ' + str(value))


    def build_index(self):
        '''
        Method for rebuilding the users/items index. If the set_item_label / set_user_label setter method
        has been used, this method is optional.
        '''
        self.users_index = {}
        self.items_index = {}

        for i, user in enumerate(self.users_label):
            self.users_index[user] = i

        for i, item in enumerate(self.items_label):
            self.items_index[item] = i


    def similar_user_knn(self, user_index, item_index, k = 5):
        '''
        Find similar user using knn approach in the full space
        '''
        new_user_coord = self.relationship_matrix[user_index]
        similarities = np.array([])

        for coord in self.relationship_matrix:
            similarity = self._euclidean_distance(coord, new_user_coord)
            similarities = np.r_[similarities, similarity]

        pruned_similarities = np.where((similarities > similarity_threshold) == True)[0]

        similar_item = {}
        for item in pruned_similarities:
            similar_item[item] = similarities[item]

        sorted_similar_items = sorted(similar_item.iteritems(), key=itemgetter(1), reverse = True)

        if not similarities_output:
            return [i[0] for i in sorted_similar_items]
        else:
            return [i[0] for i in sorted_similar_items], [i[1] for i in sorted_similar_items]


    def similar_users_cached(self, user_index, nbr_recommendations = 2, similarity_threshold = False, similarities_output = False):
        '''
        Return similar users to a given user using the precomputed users_similarities_values array
        '''

        ids = np.where(self.users_similarities_values[user_index,:] > similarity_threshold)[0]
        sorted_similar_items = self.users_similarities_sorted_id[user_index, ids]
        similarities = self.users_similarities_values[user_index, ids]

        nbr_results = len(sorted_similar_items)

        if nbr_recommendations == 'All':
            nbr_recommendations = nbr_results - 1

        if not similarities_output:
            return sorted_similar_items[0:nbr_recommendations].astype(int)
        else:
            return sorted_similar_items[0:nbr_recommendations].astype(int), similarities


    def find_user_top_match(self, user_index, nbr_recommendations = 5, k = 20, k_min = 10, sim = 0.15, rating_normalisation = True):
        '''
        Compute all the item's rating for a given user and output the most relevant
        '''
        user_ratings = np.zeros(self.nbr_items)
        already_rated = find(self.relationship_matrix_csc[:,user_index])[0]
        already_rated = np.r_[already_rated,user_index]

        for i, rating in enumerate(user_ratings):
            if i not in already_rated:
                try:
                    rating = self.predict_rating_userbased(user_index, i, k, k_min, 'All', sim, rating_normalisation)
                except:
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

        return [int(i[0]) for i in sorted_top_results], [i[1] for i in sorted_top_results]


    def most_popular_items(self, n = 10):
        try:
            return self.sorted_items_by_count[0:n]
        except TypeError:
            self.initialize_rated_feedback()
            items_count = self.N.astype(np.int16).sum(axis=0)
            self.sorted_items_by_count = np.argsort(items_count)[:,::-1].tolist()[0]
            return self.sorted_items_by_count[0:n]


    def predict_rating_userbased(self, user_index, item_index, k = 20, k_min = 5, max_iterations = 'All', similarity_threshold = False, rating_normalisation = True):
        '''
        Guess a new rating using the most similar user (user-based approach)
        Very naive method base on nearest neighbor in a dimensionly-reduced space

            * user_index : Index of the user
            * item_index : Index of the item
            * k : Number of neighbor to use
            * max_iterations : Maximum number of item use while searching for neighbors

        '''
        normalisation = 1.0
        rating = self.relationship_matrix[user_index, item_index]
        items_ratings = self.relationship_matrix_csc[item_index, :].toarray()

        if not rating:
            if self.warmedup:
                most_similar_users, similarities = self.similar_users_cached(user_index, max_iterations, similarity_threshold, True)
            else:
                most_similar_users, similarities = self.similar_users(user_index, max_iterations, similarity_threshold, True)

            try:
                rated = items_ratings[:,most_similar_users].nonzero()[1]
                ids = [most_similar_users[i] for i in rated]
                rated_ratings = items_ratings[:,ids][0]
                #rated, _, rated_ratings = find(self.relationship_matrix[most_similar_users, item_index])
                if rating_normalisation:
                    normalisation = sum([similarities[item] for item in rated[0:k]])
            except:
                raise Error("No similar neighbors found: can't make a prediction")

            real_k = rated[0:k].shape[0]

            if real_k == 0 or real_k < k_min:
                raise Error("No rated neighbors found: can't make a prediction")
            #elif real_k != k:
            #    self.logger.info('Insufficient number of neighbors for using k = ' + str(k) + ': using k = ' + str(real_k) + ' instead')

            for neighbor, neighbor_rating in zip(rated[0:k], rated_ratings[0:k]):
                rating += similarities[neighbor] * neighbor_rating / normalisation

        if self.data_normalized:
            rating = rating + self.users_bias[user_index]

        return rating


    def predict_rating_prototype(self, user_index, item_index):
        '''
        Need to be use with a missing-value awared svd (simon funk style)
        '''
        rating = self.relationship_matrix[user_index, item_index]

        if not rating:
            rating = np.dot(self.svd_v.T[user_index,:], self.svd_u[:,item_index])
        
        if self.data_normalized:
            rating = rating + self.users_bias[user_index]
        
        return rating
    
    
    def predict_rating_itembased(self, user_index, item_index, k = 20, k_min = 5, max_iterations = 250):
        '''
        Guess a new rating using the most similar items (item-based approach)
        Very naive method base on nearest neighbor in a dimensionly-reduced space
        
            * user_index : Index of the user
            * item_index : Index of the item
            * k : Number of neighbor to use
            * max_iterations : Maximum number of item use while searching for neighbors
            
        '''
        
        rating = self.relationship_matrix[user_index, item_index]
        
        if not rating:
            if self.warmedup:
                most_similar_items = self.items_similarities_sorted_id[item_index,:].todense().tolist()[0]
                similarities = self.items_similarities_values[item_index,:].todense().tolist()[0]
            else:
                most_similar_items, similarities = self.similar_items(user_index, max_iterations, similarity_threshold, True)

            try:
                rated, _, rated_ratings = find(self.relationship_matrix[user_index, most_similar_items])
                normalisation = sum([similarities[item] for item in rated[0:k]])
            except:
                raise Error("No similar neighbors found: can't make a prediction")

            real_k = rated[0:k].shape[0]
            
            if real_k == 0 or real_k < k_min:
                raise Error("No rated neighbors found: can't make a prediction")
            elif real_k != k:
                self.logger.info('Insufficient number of neighbors for using k = ' + str(k) + ': using k = ' + str(real_k) + ' instead')


            for neighbor, neighbor_rating in zip(rated[0:k], rated_ratings[0:k]):
                rating += similarities[neighbor] * neighbor_rating / normalisation

        return rating
            
    
    def _get_new_item_id(self):
        '''
        Assigns a unique id to a new item
        @todo: do it for real
        '''
        new_id = len(self.users_label)
        self.items_label.append('item'+str(new_id))        
        return new_id
   

    def _get_new_user_id(self):
        '''
        Assigns a unique id to a new user
        @todo: do it for real        
        '''
        new_id = len(self.users_label)
        self.users_label.append('user'+str(new_id))
        return new_id

    
    def add_item(self, label):
        '''
        Add a new item
        '''
        dim_y = self.relationship_matrix.shape[1]
        new_row = np.zeros(dim_y)
        self.relationship_matrix = np.c_[self.relationship_matrix.T, new_row].T
        new_id = self._get_new_item_id()
        self.items_index[label] = new_id
        return new_id    
  

    def add_user(self, label):
        '''
        Add a new user
        '''
        dim_x = self.relationship_matrix.shape[0]
        new_col = np.zeros(dim_x)
        self.relationship_matrix = np.c_[self.relationship_matrix, new_col]
        new_id = self._get_new_user_id()
        self.users_index[label] = new_id
        return new_id
            

    def set_item_raw(self, user_index, items_array):
        '''
        Assigns a set of values for each item for a given user
        '''
        if isinstance(items_array, np.ndarray):
           self.relationship_matrix[user_index, :] = items_array.astype(float)
        else:
            raise Error
        
        
    def set_user_label(self, user_index, label):
        '''
        Setter for the user label. The items_index is also updated in the process.
        ''' 
        del self.users_index[self.users_label[user_index]]
        self.users_index[label] = user_index
        self.users_label[user_index] = label
    
        
    def set_item_label(self, item_index, label):
        '''
        Setter for the item label. The items_index is also updated in the process.
        '''
        del self.items_index[self.items_label[item_index]]
        self.items_index[label] = item_index
        self.items_label[item_index] = label
    

    def get_ratings(self, randomize_order = False):
        '''
        Construct two dense array of the all the ratings
        '''
        nbr_ratings = find(self.relationship_matrix)[2].shape[0]
        ratings = np.zeros(nbr_ratings, dtype = np.float64)
        ratings_index = np.zeros([nbr_ratings,2], dtype = np.int32)
        
        for i, (user_index, feature_index, rating) in enumerate(self.ratings_iterator()):
            ratings_index[i] = [int(user_index), int(feature_index)]
            ratings[i] = rating
        
        index = np.arange(nbr_ratings)
        if randomize_order:
            np.random.shuffle(index)
            
        return ratings_index[index], ratings[index]


    def get_items_subset(self, count = 100, method = 'random'):
        '''
        Construct two dense array of a random sampling
        of 'count' items.

            * count: Number of ratings in the subset
            * method: Method uses for generating the subset ['random', 'last']
        '''
        #np.random.seed()
        ids = np.arange(self.nbr_items)
        np.random.shuffle(ids)
        return ids[0:count]


    def get_ratings_dense(self, randomize_order = False):
        '''
        Construct a single dense array of the all the ratings
        '''
        nbr_ratings = find(self.relationship_matrix)[2].shape[0]
        ratings = np.zeros([nbr_ratings,3], dtype = np.int32)
        
        for i, (user_index, feature_index, rating) in enumerate(self.ratings_iterator()):
            ratings[i] = [int(user_index), int(feature_index), rating]
        
        index = np.arange(nbr_ratings)
        if randomize_order:
            np.random.shuffle(index)
            
        return ratings[index]


    @property
    def svd_full_s(self):
        '''
        Recover the diagonalized S matrix
        @todo: use np.diag(s) instead
        '''
        s_size = len(self.svd_s)
        if s_size > 0:
            self._svd_full_s = np.zeros([s_size, s_size])
            
            for i in range(s_size):
                self._svd_full_s[i,i] = self.svd_s[i]
        
        return self._svd_full_s
    
    
    def _compute_svd(self, normalize_data = False):
        self.logger.info('Computing the Singular Value Decomposition of the relation matrix')
     
        if normalize_data:
            #self.data_normalization_item()
            self.data_normalization()
            
        self.relationship_matrix_csc = self.relationship_matrix.T.tocsc()
        self.svd_u, self.svd_s, self.svd_v = sparsesvd(self.relationship_matrix_csc, self.dimensionality)
        
    
    def _compute_items_similarities(self, k):
        '''
        Precomputes similarities between items
        '''
        self.logger.info('Computing similarities between items')
        self.items_similarities_sorted_id = lil_matrix((self.nbr_items, k))
        self.items_similarities_values = lil_matrix((self.nbr_items, k))
        
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


    def warmup(self, k = False, normalize_data = True):
        '''
        Precompute somes stuff and cache it for acceleration purposes
        '''
        if not k:
            k = self.nbr_users - 1
         
        self._compute_svd(normalize_data)
        self._compute_users_similarities(k) 
        self.warmedup = True
        self.logger.debug("Warmup done")

        
        
    def _euclidean_distance(self, A, B):
        '''
        Return the euclidean distance between two point    
        '''
        if isinstance(A, np.ndarray):
            return np.linalg.norm(A - B)
        else:
            return np.linalg.norm(A.todense() - B.todense())


    def _cosine_similarity(self, A, B):
        '''
        Compute the similarity between two vector using the cosine similarity method
        '''
        inner_product =  np.inner(A, B)
        if inner_product != 0:
            return (inner_product / (np.linalg.norm(A)*np.linalg.norm(B)))
        else:
            return 0.0


    def _cosine_similarity_log(self, A, B):
        '''
        Compute the similarity between two vector using the cosine similarity method
        '''
        inner_product =  np.inner(A, B)
        if inner_product != 0:
            return np.log( 1.0 + (inner_product / (np.linalg.norm(A)*np.linalg.norm(B))) )
        else:
            return 0.0

        
    def _pearson_similarity(self, A, B):
        return scipy.stats.pearsonr(A, B)[0]
    
    
    def _user_coordinates(self, user_index):
        return self.svd_v[:,user_index].T
    
    
    def similar_users_new(self, items_array, nbr_recommendations = 2):
        '''
        Find users fitting an external items array
        '''
        new_user_coord = self.new_user_coordinates(items_array)
        similarities = np.array([])
        
        for coord in self.svd_v.T[:,0:self.dimensionality]:
            similarity = self._cosine_similarity(coord, new_user_coord)
            similarities = np.r_[similarities, similarity]
        
        sorted_index = similarities.argsort()
        users_index = sorted_index[-nbr_recommendations:]
        
        return users_index
    
    
    def similar_users(self, user_index, nbr_recommendations = 2, similarity_threshold = False, similarities_output = False, method = 'pearson'):
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


    def similar_items_full(self, item_index, nbr_recommendations = 2, similarity_threshold = False, similarities_output = False, method = 'cosine'):
        '''
        Find items similar to a given existing item by looking at the neighbors in the full non-reduced space
            * item_index : id of the user
            * nbr_recommendations : Numbers of recommendations in the output
            * similarity_threshold : If specified, the neighbors with similarity < similarity_threshold
              will be excluded
            * similarities_output : If true, the actual value of the similarities will be included 
              in the result
            * method: Method used for computing the similarity ('cosine', 'pearson' or 'euclidean')
            
        '''
        if self.relationship_matrix_csc == None:
            self.relationship_matrix_csc =  self.relationship_matrix.tocsc()

        item_coord = self.relationship_matrix_csc.getcol(item_index).toarray()
        similarities = np.array([])
        
        '''
        Assign the method use for computing similarity between vector
        '''
        compute_similarity  = {
          'cosine':     lambda: self._cosine_similarity,
          'pearson':    lambda: self._pearson_similarity,
          'euclidean' : lambda : self._euclidean_distance,
        }[method]()
        
        for i in range(self.nbr_users):
            coord = self.relationship_matrix_csc.getcol(i).toarray()
            similarity = compute_similarity(coord.T[0], item_coord.T[0])
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
   

    
    def similar_items_by_label(self, item_label, nbr_recommendations = 2, similarity_threshold = False, similarities_output = False, method = 'cosine'):
        item_index = self.items_index[item_label]
        
        if not similarities_output:
            indexes = self.similar_items(item_index, nbr_recommendations, similarity_threshold, similarities_output, method)
        else:
            indexes , sim  = self.similar_items(item_index, nbr_recommendations, similarity_threshold, similarities_output, method)

        labels = [self.items_label[i] for i in indexes]

        if not similarities_output:
            return labels
        else:
            return labels, sim
        

    def similar_items(self, item_index, nbr_recommendations = 2, similarity_threshold = False, similarities_output = False, method = 'cosine'):
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
        item_coord = self.svd_u[:,item_index]
        similarities = np.array([])
        
        '''
        Assign the method use for computing similarity between vector
        '''
        compute_similarity  = {
          'cosine':     lambda: self._cosine_similarity,
          'pearson':    lambda: self._pearson_similarity,
          'euclidean' : lambda : self._euclidean_distance,
        }[method]()
        
        for coord in self.svd_u.T:
            similarity = compute_similarity(coord, item_coord)
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
        

    def similar_items_knn(self, item_index, nbr_recommendations = 2, label = False):
        '''
        Find items similar to a given existing item
        -- Extremely non-efficient (for dev purposes only) --
        '''
        
        self.logger.info('Searching for similar item')

        item_coord = self.relationship_matrix[:,item_index]
        similarities = np.array([])
        
        for coord in self.relationship_matrix.T:
            similarity = self._euclidean_distance(coord, item_coord)
            similarities = np.r_[similarities, similarity]
        
        sorted_index = similarities.argsort()
        users_index = sorted_index[-nbr_recommendations-1:]
        
        result_id = users_index[0:nbr_recommendations][::-1]
        
        if label:
            return self._find_item_label(result_id.tolist())
        else:
            return result_id
        
        
    def similar_user_knn(self, user_index, nbr_recommendations = 2, similarity_threshold = 0.25, similarities_output = False):
        new_user_coord = self.relationship_matrix[user_index]
        similarities = np.array([])
        
        for coord in self.relationship_matrix:
            similarity = self._euclidean_distance(coord, new_user_coord)
            similarities = np.r_[similarities, similarity]
        
        pruned_similarities = np.where((similarities > similarity_threshold) == True)[0]
        
        similar_item = {}
        for item in pruned_similarities:
            similar_item[item] = similarities[item]
        
        sorted_similar_items = sorted(similar_item.iteritems(), key=itemgetter(1), reverse = True)
        
        if not similarities_output:
            return [i[0] for i in sorted_similar_items]
        else:
            return [i[0] for i in sorted_similar_items], [i[1] for i in sorted_similar_items]
        
    
    def recommend_item_to_user(self, user_index, nbr_recommendations = 2, label = False):
        '''
        Non quantitative recommendation. Place the user in the feature space and return the nearest
        items (?!?!?)
        '''
        user_coord = self.svd_v.T[user_index,0:self.dimensionality]
        similarities = np.array([])
        
        for coord in self.svd_u.T:
            similarity = self._cosine_similarity(coord, user_coord)
            similarities = np.r_[similarities, similarity]
        
        sorted_index = similarities.argsort()
        users_index = sorted_index[-nbr_recommendations-1:]
        
        result_id = users_index[0:nbr_recommendations][::-1]
        
        if label:
            return self._find_item_label(result_id.tolist())
        else:
            return result_id
        
            
    def recommend_item_to_external(self, ratings, nbr_recommendations = 2, label = False):
        '''
        Non quantitative recommendation. Place a set of items in the feature space and 
        return the nearest items.
        '''
        feature_array = np.zeros(self.svd_u.shape[1])
        for rating in ratings:
            feature_array[rating['feature_id']] = rating['rating']
            
        user_coord = self.new_user_coordinates(feature_array)
        similarities = np.array([])
        
        for coord in self.svd_u.T:
            similarity = self._cosine_similarity(coord, user_coord)
            similarities = np.r_[similarities, similarity]
        
        sorted_index = similarities.argsort()
        users_index = sorted_index[-nbr_recommendations-1:]
        
        result_id = users_index[0:nbr_recommendations][::-1]
        
        if label:
            return self._find_item_label(result_id.tolist())
        else:
            return result_id

    
    def new_user_coordinates(self, items_array):
        '''
        Return the coordinates in the reduced spaced for a given set of features
        --- Highly inefficient (for dev only) --- 
        '''
        if self.svd_u == None:
            self._compute_svd()
        
        a = np.dot(items_array, self.svd_u.T)
        b = np.linalg.inv(self.svd_full_s[0:self.dimensionality,0:self.dimensionality])
                
        return np.dot(a,b)
