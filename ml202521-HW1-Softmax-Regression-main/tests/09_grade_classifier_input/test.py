import numpy as np
from common import assert_ndarray_equal, load_test_data
from os.path import abspath, dirname, join
from solution import Trainer


def test_classifier():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))
  trainer = Trainer()
  trainer.train_model(X_train, y_train)
  y_pred =  trainer.predict_model(X_test)
  accuracy = np.mean(y_pred==y_test)
  with open('logs.txt', 'w') as file:
    print(f'{accuracy:.3f}', file=file)
  assert accuracy > 0.0
      
      
