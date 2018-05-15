from dnn.model.speechModel import SpeechModel
from dnn.utils.parser import test_config

config = test_config("cnn-trad-pool2")
config['n_labels'] = 10
model = SpeechModel(config)
print(model)
