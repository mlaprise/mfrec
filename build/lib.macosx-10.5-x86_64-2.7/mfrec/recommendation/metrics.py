'''
    Evaluation function for testing the accuracy of the predictions / recommendations
    
    Created on October 18th, 2011
    
    @author: Martin Laprise
     
'''

import numpy as np


def shannon_entropy(self, recommender):
    '''
    Shannon Entropy for items space coverage
    '''
    pass

def test_predict_rating2(MovieLensRecommender, u1_test, mi, nbr_samples = 10, predictor = 'predict_rating', verbose = False):
    real_rating = np.array([])
    predicted_rating = np.array([])
    count = 0
    for i, rating in enumerate(u1_test[0:nbr_samples]):
        try:
            predicted_rating = np.r_[predicted_rating, getattr(MovieLensRecommender, predictor)( mi[int(rating[1])], int(rating[0]) )]

            real_rating = np.r_[real_rating, rating[2]]
            if verbose:
                print 'Prediction ' + str(i) + ': Predicted = ' + str(predicted_rating[-1]) + ', Real = ' + str(real_rating[-1])
            
            count += 1
        except (Error, KeyError):
            pass
        
    all_errors = real_rating - predicted_rating
    valid_id = np.where(np.isnan(all_errors) == False)[0]
    errors = all_errors[valid_id]
    
    rmse = np.sqrt(pow(abs(errors), 2).mean())
    
    print 'Predictor: ' + str(predictor)
    print '\nNumber of succesful rating: ' + str(len(abs(errors))) + '/' + str(nbr_samples)
    print 'Mean abs. error: ' + str(abs(errors).mean())
    print 'Variance of the error: ' + str(abs(errors).var())
    print 'Mean root mean square error: ' + str(rmse)
    print 'NMAE: ' + str(abs(errors).mean() / 1.6) + '\n\n'
    
    return rmse, errors


def test_predict_rating(recommender, u_test, nbr_samples = 10, verbose = False, predictor = 'predict_rating'):
    '''

    '''
    real_rating = np.array([])
    predicted_rating = np.array([])
    
    for i, rating in enumerate(u_test[0:nbr_samples]):
        try:
            predicted_rating = np.r_[predicted_rating, getattr(recommender, predictor)( int(rating[1]), int(rating[0]) )]
                        
            real_rating = np.r_[real_rating, rating[2]]
            if verbose:
                print 'Prediction ' + str(i) + ': Predicted = ' + str(predicted_rating[-1]) + ', Real = ' + str(real_rating[-1])

        except Error:
            pass
        
    all_errors = real_rating - predicted_rating
    valid_id = np.where(np.isnan(all_errors) == False)[0]
    errors = all_errors[valid_id]
    abs_errors = abs(errors) 
    rmse = np.sqrt(pow(abs(errors), 2).mean())

    print '\nNumber of succesful rating: ' + str(len(abs_errors)) + '/' + str(nbr_samples)
    print 'Mean abs. error: ' + str(abs_errors.mean())
    print 'Variance of the error: ' + str(abs_errors.var())
    print 'Mean root mean square error (RMSE): ' + str(rmse)
    print 'NMAE: ' + str(abs_errors.mean() / 1.6)
    print 'MAE: ' + str(abs_errors.mean())  + '\n\n'
    
    return rmse, errors


def precision_recall(recommender, u_test, nbr_recommendations = 5, predictor = 'predict', verbose = False):
    '''
    Evaluate the precision-recall score for an non quantitative items recommendations process
    '''
    precision = 0.0
    recall = 0.0

    test_sample_dict = {}
    for i, rating in enumerate(u_test):
        try:
            test_sample_dict[int(rating[0])].append(int(rating[1]))
        except (KeyError, AttributeError):
            test_sample_dict[int(rating[0])] = [int(rating[1])]
     
    users_count  = 0

    recommended_set = set()

    for user_index in test_sample_dict.keys():
        try:
            recommended_set = set(recommender.find_recommended_items(user_index = user_index,
                                                                     nbr_recommendations = nbr_recommendations,
                                                                     output_label = False, predictor = predictor)[0])
            users_count += 1
        except KeyError:
            recommended_set = set()
            
        try:
            already_rated = set(test_sample_dict[user_index])
            intersection =  float(len(recommended_set.intersection(already_rated)))
            precision += intersection / nbr_recommendations
            recall += intersection / len(already_rated)
        except KeyError:
            intersection = 0
            recall = 0
        
    precision = precision / users_count
    recall = recall / users_count
    f_measure = 2 * (precision * recall) / (precision + recall)
    
    if verbose:
        print 'Precision @ ' + str(nbr_recommendations) + ' : ' + str(precision) 
        print 'Recall @ ' + str(nbr_recommendations) + ' : ' + str(recall)     
        print 'F-Measure : ' + str(f_measure)
        
    return precision, recall, f_measure


def folding_in_test(recommender, u, u_test, ratio = 0.10):
    '''
    Folding-in test 

    ### NOT COMPLETE ###
    '''
    nbr_users = recommender.nbr_users
    nbr_users_to_remove = int(nbr_users * 0.10)
    all_index = np.arange(1,nbr_users)
    np.random.shuffle(all_index)
    remove_user = all_index[0:nbr_users_to_remove]
    
    ids = {}
    for user_index in remove_user:
        ids[user_index] = np.where(u[:,0] == user_index)[0].astype(np.int32)

    ids_test = {}
    for user_index in remove_user:
        ids_test[user_index] = np.where(u_test[:,0] == user_index)[0].astype(np.int32)

    flatten_ids = np.array([])
    for user_index in remove_user:
        flatten_ids = np.r_[flatten_ids, ids[user_index]]

    flatten_ids_test = np.array([])
    for user_index in remove_user:
        flatten_ids_test = np.r_[flatten_ids_test, ids_test[user_index]]

    prune_train_set = np.delete(u,flatten_ids,0)
    prune_test_set = u_test[flatten_ids_test.astype(np.int32)]
    
    print 'Testing on the complete test set'
    test_predict_rating(recommender, u_test, nbr_samples = 20000, predictor = 'predict_logistic')

    print 'Testing on the pruned test set'
    test_predict_rating(recommender, prune_test_set.astype(np.int32), nbr_samples = 20000, predictor = 'predict_logistic')

    '''
    Train on the prune train set
    '''
    recommender.__init__(nbr_users+1, nbr_feature+1, parameters)
   
    for rating in prune_train_set:
        recommender.set_item_by_id(rating[0], rating[1], float(rating[2]))
   
    print 'Train on the prune train set'
    print '###############################'        
    recommender.train(verbose = False)
    
    print 'Testing on the complete test set'
    test_predict_rating(recommender, u_test, nbr_samples = 20000, predictor = 'predict_logistic')

    print 'Testing on the pruned test set'
    test_predict_rating(recommender, prune_test_set.astype(np.int32), nbr_samples = 20000, predictor = 'predict_logistic')
    

    '''
    Folding-in pruned users
    '''
    print 'Folding-in pruned users'
    print '###############################'
    
    #for user_index in remove_user:
    #    MovieLensRecommender.retrain_user(user_index, u[ids[user_index]].astype(np.int32)[:,0:3], verbose = False)
    
    for user_index in remove_user:
        users_ratings = u_test[ids_test[user_index].astype(np.int32)][:,2]
        users_ratings_index = u_test[ids_test[user_index].astype(np.int32)][:,1]
        if users_ratings.shape[0] > 0:
            recommender.add_user('newuser' + str(user_index), users_ratings_index, users_ratings)

    print 'Testing on the complete test set'
    test_predict_rating(recommender, u_test, nbr_samples = 20000, predictor = 'predict_logistic')

    print 'Testing on the pruned test set'
    test_predict_rating(recommender, prune_test_set.astype(np.int32), nbr_samples = 20000, predictor = 'predict_logistic')

