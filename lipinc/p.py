import numpy as np
import tensorflow as tf
import mediapipe as mp
import sklearn
import scipy

print("numpy:", np.__version__)
print("tensorflow:", tf.__version__)
print("mediapipe:", mp.__version__)
print("scipy:", scipy.__version__)
print("sklearn:", sklearn.__version__)
print("has numpy.exceptions:", hasattr(np, "exceptions"))
print("has mp.solutions:", hasattr(mp, "solutions"))
