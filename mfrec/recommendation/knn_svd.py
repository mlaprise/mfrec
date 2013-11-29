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


class KNNSVDRecommender(MFRecommender):
    '''
    
    Simple Recommender base on a Singular Value Decomposition 
    
    A Singular Value Decomposition is use for reducing the dimensionality of the space.
    The optimal dimensionality depends on the nature of the problem (aka the dataset). The optimal k value
    depends on the nature of the problem and on the sparsity on the relation matrix.
   
    Since this is pure singular value decomposition (not a regularized matrix factorisation) the item feature vector
    is not self.svd_u but self.svd_u * self.svd_s

    The class include some knn method for a hybrid svd-knn recommender
    '''
    
    PARAMETERS_INDEX = {'k' : 'k',
                        'k_min' : 'k_min',
                        'sim_threshold' : 'sim_threshold',
                        'nbr_features' : 'dimensionality'}


    def __init__(self,  nbr_users = 4, nbr_items = 6, parameters = False, filename = False):
        MFRecommender.__init__(self, nbr_users, nbr_items, filename)

        # Initialize the training parameters with the default value
        self.k = 80
        self.k_min = 2
        self.sim_threshold = 0.18
        self.dimensionality = 40

        if parameters:
            self.set_parameters(parameters)
            
        
    
    def __repr__(self):
        string = 'Simple SVD-KNN Recommendation Engine\n'
        string += 'Number of users: ' + str(self.nbr_users) + '\n'
        string += 'Number of items: ' + str(self.nbr_items) + '\n'
        string += 'Dimensionality: ' + str(self.dimensionality) + '\n'
        return string
   

    def train(self):
        self.warmup(k = self.k, normalize_data = True)
   

    def predict(self, item_index, user_index):
        ri = self.relationship_matrix[:,item_index].T.toarray()
        pu = self.svd_v[user_index,:]
        Q = self.svd_v.T
        a = np.dot(pu, ri)
        b = np.dot(a, Q)
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
            #self.data_normalization_item()
            self.data_normalization()
            
        self.relationship_matrix_csc = self.relationship_matrix.T.tocsc()
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


    def warmup(self, k = False, normalize_data = True):
        '''
        Precompute somes stuff and cache it for acceleration purposes
        '''
        if not k:
            k = self.k
         
        self._compute_svd(normalize_data)
        self._compute_users_similarities(k) 
        self.warmedup = True
        self.logger.debug("Warmup done")
    
    
    def predict_rating_userbased(self, item_index, user_index, k = 20, k_min = 5, max_iterations = 'All', similarity_threshold = False, rating_normalisation = True):
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
        feature_coord = self.svd_u[:,item_index]
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
            similarity = compute_similarity(coord, feature_coord)
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
        
        return [int(i[0]) for i in sorted_top_results], [i[1] for i in sorted_top_results]

