class Config:
    def __init__(self):
        self.dev_path = 'data/dev.csv'
        self.train_path = 'data/train.csv'
        self.max_vocab = 5000
        self.max_sent_len = 30
        self.lr = 1e-4
        self.epoch = 3
        self.batch_size = 64