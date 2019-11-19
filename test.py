from ensemble_k_means import *
from elbow_k_means import *
import time

model_path = 'weight/namuwiki-2.model'

print(elbow_k_means("강아지", model_path=model_path))
