import numpy as np
import scipy
from scipy import ndimage

from lr_mindset import num_px1, num_px2, predict, d, classes

my_image = "cat_in_iran.jpg"
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten = False))
my_image = scipy.misc.imresize(image, size = (num_px1, num_px2)).reshape((1, num_px1 * num_px2 * 3)).T
my_prediction_image = predict(d["w"], d["b"], my_image)

print("y = " + str(np.squeeze(my_prediction_image)) + ", you algorithm predicts a \"" + classes[int(np.squeeze(my_prediction_image)),].decode("utf-8") + "\" picture.")