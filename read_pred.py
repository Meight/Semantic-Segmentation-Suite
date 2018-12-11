import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

from utils import utils
from utils.utils import save_image

image = 'C:/Work/Python/datasets/VOC2012/JPEGImages/2007_000256.jpg'
loaded_image = utils.load_image(image)
cv2.save("%s_pred.png"%("test"), cv2.cvtColor(np.uint8(loaded_image), cv2.COLOR_RGB2BGR))



plt.figure()
plt.imshow(loaded_image)
plt.show()