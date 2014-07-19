import networkx as nx

MAX_NBR_RELATED = 50

class SimilarityGraph(object):
    '''
    Generate a similarity Graph using a recommender
    '''
    def __init__(self, recommender, nodes_attributes = None):
        self.recommender = recommender
        # Initialize the graph
        self.similarity_graph = nx.Graph()
        self.nodes_attributes = nodes_attributes

    def set_recommender(self, recommender):
        self.recommender = recommender

    def add_node(self, n):
        if not self.similarity_graph.has_node(n):
            self.similarity_graph.add_node(n)
            if self.nodes_attributes:
                for key, value in self.nodes_attributes.iteritems():
                    self.similarity_graph.node[n][key] = value[n]

    def add_edge(self, n1, n2, weight=None):
        if not self.similarity_graph.has_edge(n1, n2):
            self.similarity_graph.add_edge(n1, n2)
            self.similarity_graph[n1][n2]['weight'] = weight
        else:
            self.similarity_graph[n1][n2]['weight'] += weight

    def build_graph(self, min_score = 0.98):

        for node_title, node_id in self.recommender.items_index.iteritems():
            self.add_node(node_title)
            related_nodes, related_scores = self.recommender.similar_items_by_label(node_title, MAX_NBR_RELATED, similarity_threshold = min_score, similarities_output=True)
            for related in zip(related_nodes, related_scores):
                self.add_edge(node_title, related[0], float(related[1]))
                print "%s --%s--> %s" % (node_title, related[1], related[0])

    def write_graph(self, name = "similarity_graph.graphml"):
        nx.write_graphml(self.similarity_graph, name)
