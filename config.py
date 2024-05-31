import configparser

train_config = configparser.ConfigParser()
train_config.read('train.ini', 'utf8')

epoch_num = train_config.getint('train_parameters', 'epochs')
learning_rate = train_config.getfloat('train_parameters', 'learning_rate')
batch_size = train_config.getint('train_parameters', 'batch_size')

train_dataset_path = train_config.get('dataset', 'train_dataset_path')
test_dataset_path = train_config.get('dataset', 'test_dataset_path')
dataset_path = train_config.get('dataset', 'dataset_path')
dataset_root = train_config.get('dataset', 'dataset_root')
dataset_size = train_config.getint('dataset', 'dataset_size')

crop_size = train_config.getint('image_settings', 'crop_size')