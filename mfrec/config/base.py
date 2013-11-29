'''
    Crowdbase settings module
    
    Created on May 24th, 2011
    
    @author: Martin Laprise
        
'''

import os
import datetime
from com.crowdbase.config.environment import get_hostname, get_system_platform

home_folder = os.getenv('HOME')
local = True if get_system_platform() == 'win32' or get_hostname().find('.local') > 0 else False


'''
Location and format of the log file and other paths
'''
LOG_INTERVAL = 1
LOG_BACKUPCOUNT = 10
if local:
    LOG_DIR = home_folder + '/log/'
    SRC_DIR = home_folder + '/crowdbase-analytics/'
    DATASETS_HOME = home_folder + '/datasets/'
    MLCOMP_HOME = home_folder + '/datasets/mlcomp/'    
    CLASSIFIER_DATA_HOME = '/Users/mlaprise/classifier_data/'
    SSH_KEYS = home_folder + '/.ssh/'
else:
    LOG_DIR = '/home/ubuntu/log/'
    SRC_DIR = '/home/ubuntu/crowdbase-analytics/'
    MLCOMP_HOME = '/home/ubuntu/datasets/mlcomp/'
    DATASETS_HOME = '/home/ubuntu/datasets/'    
    CLASSIFIER_DATA_HOME = '/home/ubuntu/classifier_data/'
    

LOG_FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d (%(process)d) %(message)s'


'''
Config of the databases
'''
if local:
    DATABASE = {'ip' : 'localhost','name' : 'crowdbase_development'}
    CBR_DATABASE = {'ip':'localhost', 'name' : 'crowdbase_recommender'}
    QUEUE_DATABASE = {'ip':'localhost', 'db' : 0, 'port' : 6379}
    
else:
    DATABASE = {'ip' : ['10.126.50.108','10.118.230.219'], 'name' : 'crowdbase_development', 'replicaset' : ''}
    CBR_DATABASE = {'ip':['10.126.50.108','10.118.230.219'], 'name' : 'crowdbase_recommender', 'replicaset': ''}
    QUEUE_DATABASE = {'ip':'localhost', 'db' : 0, 'port' : 6379}
    

INSTANCES = [('mirego','crowdbase'),
             ('crowdbase','crowdbase'),
             ('duproprio','crowdbase'),
             ('testeurs','crowdbase'),
             ('','famille20'),
             ('founderfuel','crowdbase'),
             ('', 'macquebec-entraide'),
             ('qi', 'crowdbase')]

CONCEPTS_INSTANCES = [('crowdbase', 'crowdbase'),
                      ('lp','crowdbase')] 

'''
Daemons settings
'''
if local:
    DELAY_BETWEEN_RECOMMENDATION_REFRESH = 15
    DELAY_BETWEEN_TICKER_REFRESH = 10   
    API_DAEMON_DELAY_BETWEEN_REFRESH = 15
    RECOMMENDER_API_PORT = 30080
    LANGUAGE_API_PORT = 30090
    CLASSIFIER_API_PORT = 30100
    CONCEPT_API_PORT = 30200    
    LANGUAGE_DEMO_PORT = 40080
    SEARCH_ADMIN_PORT = 30300         
    
else:
    DELAY_BETWEEN_RECOMMENDATION_REFRESH = 900
    DELAY_BETWEEN_TICKER_REFRESH = 60        
    API_DAEMON_DELAY_BETWEEN_REFRESH = 300
    RECOMMENDER_API_PORT = 30080
    LANGUAGE_API_PORT = 30090    
    CLASSIFIER_API_PORT = 30100
    CONCEPT_API_PORT = 30200     
    LANGUAGE_DEMO_PORT = 40080
    SEARCH_ADMIN_PORT = 30300     
    
    
    
USERS_GDRECOMMENDER_CONFIG = {'dim' : 20, 'nbr_recommendations' : 10}
TOPICS_GDRECOMMENDER_CONFIG = {'dim' : 20, 'nbr_recommendations' : 10}
QUESTIONS_GDRECOMMENDER_CONFIG = {'dim' : 20, 'nbr_recommendations' : 10}


USERS_RECOMMENDER_CONFIG = {'k' : 10, 'k_min' : 2, 'sim' : 0.50, 'nbr_recommendations' : 5}
TOPICS_RECOMMENDER_CONFIG = {'k' : 10, 'k_min' : 2, 'sim' : 0.50, 'nbr_recommendations' : 5}
CLASSIFIER_CONFIG = {'nbr_features' : 50000}

UPDATE_RATINGS_SYNCHRONOUSLY = True
HANDLE_SIGNALS_SYNCHRONOUSLY = True

'''
Crowdbase API
'''
CROWDBASE_API_URL = 'https://coreapi.crowdbase.com'
CROWDBASE_API_URL_LOCAL = 'http://127.0.0.1:3000'
API_ACCESS_USER = 'q_and_a'
API_ACCESS_PWD = 'sebsurfer'

'''
SOLR
'''
if local:
    SOLR_HOME = '/Users/mlaprise/apache-solr-3.5.0/example/crowdbase/'    
    #SOLR_SERVER = 'http://ec2-50-19-181-240.compute-1.amazonaws.com:8983/solr/'
    SOLR_SERVER = 'http://127.0.0.1:8983/solr/'
    STANBOL_SERVER = 'http://ec2-50-19-181-240.compute-1.amazonaws.com:8080/engines/'
    SEARCH_SERVER = 'http://127.0.0.1:8983/solr/'
    SEARCH_SCHEMA = SRC_DIR + 'com/crowdbase/search/solr_config/crowdbase/shared_schema/'
    SOLR_SERVER_MASTER = 'http://127.0.0.1:8983/solr/'
    SOLR_SERVER_SLAVE = 'http://127.0.0.1:8983/solr/'
    CONCEPTS_URL = 'http://ec2-50-19-181-240.compute-1.amazonaws.com/v1/concepts'
    GRAPH_SERVER = 'http://ec2-50-17-74-75.compute-1.amazonaws.com:7474/db/data/'
    INACTIVE_THRESHOLD = datetime.timedelta(0,15,0)
    SEARCH_ADMIN_URL = 'http://ec2-50-19-181-240.compute-1.amazonaws.com/v1/search'
    
else:
    SOLR_HOME = '/mnt/opt/solr/crowdbase/'
    SOLR_SERVER = 'http://127.0.0.1/solr/'
    SOLR_SERVER_MASTER = 'http://10.34.99.72:8983/solr/'
    SOLR_SERVER_SLAVE = 'http://10.120.227.37:8983/solr/'
    SOLR_SERVER_MASTER_HOST = 'http://10.34.99.72'
    SOLR_SERVER_SLAVE_HOST = 'http://10.120.227.37'        
    STANBOL_SERVER = 'http://10.34.43.42:8080/engines/'
    SEARCH_SERVER = 'http://127.0.0.1:8983/solr/'
    SEARCH_SCHEMA = '/home/ubuntu/crowdbase-analytics/com/crowdbase/search/solr_config/crowdbase/shared_schema/'
    CONCEPTS_URL = 'http://ec2-50-19-181-240.compute-1.amazonaws.com/v1/concepts'
    GRAPH_SERVER = 'http://ec2-50-17-74-75.compute-1.amazonaws.com:7474/db/data/'    
    INACTIVE_THRESHOLD = datetime.timedelta(3,0,0)
    SEARCH_ADMIN_URL = 'http://ec2-50-19-181-240.compute-1.amazonaws.com/v1/search'

FREEBASE_DATASETS = 'http://download.freebase.com/wex/latest/'
CLASSIFIER_CACHE_SIZE = 512
SMARTTOPIC_CACHE_SIZE = 64 

AWS_ACCESS = 'AKIAI4M2TCJ6HBZNBBWA'
AWS_SECRET = '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'

UNIFY_POST_BY_MAIL = set(['inbox', 'post'])
#### Email classifier feature extraction parameters ####
LONG_THRESHOLD = 140
FIRST_SENTENCE = 2

### Mixpanel ###
# Uncomment this for using mixpanel
# Use this token only in PRODUCTION
#mixpanel_api_key = '8c9a7e30bc9f655519189a1c9a822eda'
#mixpanel_api_secret = '3a983ab85626f6529ef3d8abd81663a3'
#mixpanel_api_token = 'aae508903d26b5743786fd758a7b8483'
