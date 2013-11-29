def optimize_k(MovieLensRecommender):
    nbr_test = 50
    score_vs_k = np.zeros(nbr_test)
    
    for k in np.arange(nbr_test):
        score_vs_k[k+1] = test_predict_rating(MovieLensRecommender, u1_test, 2000, k+1)
        

def optimize_similarity_threshold(MovieLensRecommender):
    sim_test = np.linspace(0.00, 0.99, 20)
    nbr_test = sim_test.shape[0]
    score_vs_sim = np.zeros(nbr_test)
    
    for i, sim in enumerate(sim_test):
        score_vs_sim[i] = test_predict_rating(MovieLensRecommender, u1_test, 200, 35, 10, sim)
        print score_vs_sim[i]


def optimize_dim(MovieLensRecommender):
    dim_test = np.arange(10,50,2)
    nbr_test = dim_test.shape[0]
    score_vs_dim = np.zeros(nbr_test)
    for i, dim in enumerate(dim_test):
        print 'Test dimensionality = ' + str(dim)
        MovieLensRecommender.change_dimensionality(dim)
        MovieLensRecommender.warmup(250)
        score_vs_dim[i] = test_predict_rating(MovieLensRecommender, u1_test, 1000, k = 35, k_min = 10)
        print score_vs_dim[i]

