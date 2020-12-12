import ktrain
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#use only CPU for precition

predictor = ktrain.load_predictor('distilBERT')

def get_prediction(x):

	sent = predictor.predict([x])
	return sent[0]

