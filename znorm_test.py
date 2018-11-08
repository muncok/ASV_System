import bob.learn.em
import numpy
numpy.random.seed(10)


n_clients = 10
n_scores_per_client = 1000

# Defining some fake scores for genuines and impostors
impostor_scores = numpy.random.normal(-15.5,
                                      5, (n_scores_per_client, n_clients))
genuine_scores = numpy.random.normal(0.5, 5, (n_scores_per_client, n_clients))

# Defining the scores for the statistics computation
z_scores = numpy.random.normal(-5., 5, (n_scores_per_client, n_clients))

# Z - Normalizing
z_norm_impostors = bob.learn.em.znorm(impostor_scores, z_scores)
z_norm_genuine = bob.learn.em.znorm(genuine_scores, z_scores)
