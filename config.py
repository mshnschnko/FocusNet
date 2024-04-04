import configparser

train_config = configparser.ConfigParser()
train_config.read('train.ini', 'utf8')

epoch_num = train_config.getint('train_parameters', 'epochs')
batch_size = train_config.getint('train_parameters', 'batch_size')
learning_rate = train_config.getfloat('train_parameters', 'learning_rate')
train_dataset_path = train_config.get('train_parameters', 'train_dataset_path')
test_dataset_path = train_config.get('train_parameters', 'test_dataset_path')