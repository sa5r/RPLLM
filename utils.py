class Utils:
    """Utility functions called by the model operations .
    """

    def __init__(self, time_stamp):
        """
        """

        self.time_stamp = time_stamp
        self.write_log(f'Time stamp {time_stamp}')
    
    def get_timestamp(self):
        return self.time_stamp

    def write_log(self, s, path = 'log.out', prnt = True):
        ''
        
        f = open(self.time_stamp + path , "a")
        f.write('\n' + s)
        if prnt:
            print(s)
        f.close()

    def load_relations(self, path: str):
        ''

        relations = []
        with open(path) as f:
            for line in f.readlines():
                relations.append(line.strip())
        
        print('\nRelations loaded')
        return relations

    # Reproduce
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)