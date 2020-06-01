import logging



class AttrDict(dict):
    '''
    Dictionary whose keys can be accessed as attributes
    '''
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        
    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]
    
    def __setattr(self, item, value):
        self.__dict__[item] = value
        
        
def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger










