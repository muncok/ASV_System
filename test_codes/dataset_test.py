from dnn.data.dataset import SpeechDataset
from dnn.parser import test_config
import pandas as pd
config = test_config("SimpleCNN")
df = pd.read_pickle("trials/commands/final/equal_num_102spk_enroll.pkl")
samples = SpeechDataset.read_df("/home/muncok/DL/dataset/SV_sets/speech_commands", df)
dataset = SpeechDataset(samples, "train", config)
print(dataset)
