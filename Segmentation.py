from skimage import img_as_float, img_as_ubyte, io
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
from matplotlib import pyplot as plt

#Use this code if you run this project on google colab
from google.colab import drive
drive.mount('/content/drive')

img = io.imread("drive/My Drive/Colab Notebooks/covid.jpg")
plt.imshow(img,cmap="gray")
#image available in /images folder covid.png

img = img_as_float(img)
plt.imshow(img,cmap="gray")


sigma_est = np.mean(estimate_sigma(img,multichannel=True))

denoise = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False, patch_size=5, patch_distance=3, multichannel=True)

plt.imshow(denoise,cmap="gray")


denoise_ubyte = img_as_ubyte(denoise)

plt.imshow(denoise_ubyte,cmap="gray")
#image available in /images folder covid.png

plt.hist(denoise_ubyte.flat, bins=100, range=(0,255))
#image available in /images folder histogram.png

seg1 = (denoise_ubyte <=10)
seg2 = (denoise_ubyte > 10) & (denoise_ubyte <= 80)
seg3 = (denoise_ubyte > 80) & (denoise_ubyte <=120)
seg4 = (denoise_ubyte > 120) & (denoise_ubyte <= 135)
seg5 = (denoise_ubyte > 135) & (denoise_ubyte <= 160)
seg6 = (denoise_ubyte > 160) & (denoise_ubyte <= 175)
seg7 = (denoise_ubyte > 175) & (denoise_ubyte <= 210)
seg8 = (denoise_ubyte > 210) 
plt.imshow(seg7,cmap="gray")
#image available in /images folder image1.png

all_seg = np.zeros((denoise_ubyte.shape[0],denoise_ubyte.shape[1],3))
all_seg[seg1] = (1,0,0)
all_seg[seg2] = (0,1,0)
all_seg[seg1] = (0,0,1)
all_seg[seg1] = (1,1,0)
all_seg[seg1] = (1,0,1)
all_seg[seg1] = (0,1,1)
all_seg[seg1] = (1,2,0)
all_seg[seg1] = (2,0,0)

plt.imshow(all_seg)
#image available in /images folder image2.png
