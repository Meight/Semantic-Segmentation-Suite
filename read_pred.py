import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

image = Image.open('C:/Work/Python/temp/resnet/2010_000906_pred.png')
image = np.asarray(image)

plt.figure()
plt.imshow(image)
plt.show()