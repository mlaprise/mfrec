import numpy as np
import time
import matplotlib.pyplot as plt
import codecs

from mfrec.recommendation.metrics import test_predict_rating, precision_recall
from mfrec.recommendation.wrmf import WRMFRecommender
from mfrec.graph.similarity_graph import SimilarityGraph



if __name__ == '__main__':
    import os
    home_folder = os.getenv('HOME')
    #datasets = ['u1', 'u2', 'u3', 'u4', 'u5']
    datasets = ['u1']

    for dataset in datasets:

        print 'Dataset: ' + str(dataset)

        u = np.loadtxt(home_folder + '/datasets/ml-100k/' + dataset + '.base')
        unique_user = list(set(u[:,0].tolist()))
        nbr_user = len(unique_user)
        unique_movie = list(set(u[:,1].tolist()))
        nbr_feature = max(unique_movie)

        parameters = {'nbr_epochs': 30,
                      'feature_init': 0.1,
                      'regularization_model': 0.015,
                      'nbr_features': 20,
                      'neighborhood' : 1500} 

        '''
        Instantiate a CrowdBaseRecommender
        '''
        MovieLensRecommender = WRMFRecommender(nbr_user+1, nbr_feature+1, parameters)

        for rating in u:
            MovieLensRecommender.set_item_by_id(rating[0], rating[1], 1.0)


        '''
        Set label
        '''
        for line in codecs.open(home_folder + '/datasets/ml-100k/u.item', 'r', 'latin-1'):
            (id,title)=line.split('|')[0:2]
            try:
                MovieLensRecommender.set_item_label(int(id), title)
            except KeyError:
                pass

        '''
        Test on the test data set
        '''
        u_test = np.loadtxt(home_folder + '/datasets/ml-100k/' + dataset + '.test')
        MovieLensRecommender.train(verbose = True, handle_bias = True)
        print precision_recall(MovieLensRecommender, u_test, nbr_recommendations = 5, verbose = True)

        # Get movies similar to Toy Story
        movie = 'Terminator 2: Judgment Day (1991)'
        similars = MovieLensRecommender.similar_items_by_label(movie, 10, method='cosine')
        print "------------"
        print similars

        # Build similarity graph
        sg = SimilarityGraph(MovieLensRecommender)
        sg.build_graph()
        sg.write_graph()
