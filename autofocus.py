import cv2
import numpy as np
import time

images = [cv2.GaussianBlur(np.random.randint(0, 256, (1640, 1232, 3), dtype='uint8'), (15, 15), i-15) for i in range(50)]
# im = cv2.randn((672, 672, 3), mean=0, stddev=1)
print(images[0].shape)


start = time.time()

max_laplacian = -1
focused_img_idx = 0
for idx, image in enumerate(images):
    # grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(image,  cv2.CV_64F).var()
    if laplacian > max_laplacian:
        max_laplacian = laplacian
        focused_img = image
        focused_img_idx = idx

end = time.time()

print(focused_img_idx)
print('avg time sharpness = ', (end-start)*1000/len(images))



start = time.time()

max_laplacian = -1
focused_img_idx = 0
for idx, image in enumerate(images):
    # grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    c_max = np.amax(image)
    c_min = np.amin(image)
    laplacian = (c_max - c_min) / (c_max + c_min)
    # laplacian = image.var()
    if laplacian > max_laplacian:
        max_laplacian = laplacian
        focused_img = image
        focused_img_idx = idx

end = time.time()

print(focused_img_idx)
print('avg time contrast 1 = ', (end-start)*1000/len(images))



start = time.time()

max_laplacian = -1
focused_img_idx = 0
for idx, image in enumerate(images):
    # grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # c_max = image.max()
    # c_min = image.min()
    # laplacian = (c_max - c_min) / (c_max + c_min)
    laplacian = image.var()
    if laplacian > max_laplacian:
        max_laplacian = laplacian
        focused_img = image
        focused_img_idx = idx

end = time.time()

print(focused_img_idx)
print('avg time contrast 2 = ', (end-start)*1000/len(images))