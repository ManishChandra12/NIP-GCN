class CONFIG(object):
    def __init__(self):
        super(CONFIG, self).__init__()
        
        self.dataset = 'M10'  # can be 'covid', 'covid_title', 'M10' or 'dblp'

        self.model = 'gcn'
        self.train_size = 0.7
        self.doc_doc_edge = True  # add doc-doc citation links?
        self.num_neg_samples = 7  # number of negative samples
        self.neg_loss = True  # whether to include L2 loss
        self.learning_rate = 0.02  # Initial learning rate
        self.epochs = 200  # Maximum number of epochs to train
        self.hidden1 = 200  # Number of units in hidden layer 1
        self.dropout = 0.5  # Dropout rate (1 - keep probability)
        self.early_stopping = 10  # Tolerance for early stopping (# of epochs)
        self.bins = 10  # number of bins for histogram mapping
        self.hist_map = 'lch'  # change to 'nh' for normalized histogram mapping

        if self.dataset == 'M10':
            self.lamb = 0.004
            self.node2vec = True  # whether to use node2vec or just the one-hot representation
            self.node2vec_interaction_initialize = True  # whether to compute interaction based on node2vec embeddings or use node2vec as is (instead of one-hot)
        elif self.dataset == 'dblp':
            self.lamb = 0.001
            self.node2vec = True
            self.node2vec_interaction_initialize = True
        elif self.dataset == 'covid_title':
            self.lamb = 0.0015
            self.node2vec = True
            self.node2vec_interaction_initialize = True
        elif self.dataset == 'covid':
            self.lamb = 0.0015
            self.node2vec = True
            self.node2vec_interaction_initialize = True
