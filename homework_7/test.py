# %%

import numpy as np
from numpy.matlib import repmat
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import cv2

# %%

video_file = "road_video.MOV"
videoCapture = cv2.VideoCapture(video_file)

# 视频帧数，分辨率
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# %%

# 指定写视频的格式，mp4
videoWriter = cv2.VideoWriter('img/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), (fps / 10), size)
videoWriter.isOpened()

# %%

# 读帧
success, frame = videoCapture.read()


# %%


# %%


# %%

def kmeans(data, n_cl, verbose):
    n_samples = data.shape[0]
    centers = data[np.random.choice(range(n_samples), size=n_cl)]
    old_labels = np.zeros(shape=n_samples)

    while True:
        distances = np.zeros(shape=(n_samples, n_cl))
        for c_idx, c in enumerate(centers):
            distances[:, c_idx] = np.sum(np.square(data - repmat(c, n_samples, 1)), axis=1)
        new_labels = np.argmin(distances, axis=1)

        for l in range(0, n_cl):
            centers[l] = np.mean(data[new_labels == l], axis=0)

        if verbose:
            fig, ax = plt.subplots()
            ax.scatter(data[:, 0], data[:, 1], c=new_labels, s=40)
            ax.plot(centers[:, 0], centers[:, 1], 'r*', markersize=20)
            plt.waitforbuttonpress()
            plt.close()

        if np.all(new_labels == old_labels):
            break

        old_labels = np.copy(new_labels)
    return new_labels


# %%
count = 0
while success and count < 10:
    img = np.float32(frame)
    h, w, c = img.shape

    row_indexes = np.arange(0, h)
    col_indexes = np.arange(0, w)
    coordinates = np.zeros(shape=(h, w, 2))
    coordinates[..., 0] = normalize(repmat(row_indexes, w, 1).T)
    coordinates[..., 1] = normalize(repmat(col_indexes, h, 1))

    data = np.concatenate((img, coordinates), axis=-1)
    data = np.reshape(data, newshape=(w * h, 5))
    labels = kmeans(data, n_cl=3, verbose=False)
    frame = (np.reshape(labels, (h, w)) * 120).astype('u1')

    IMG_OUT = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    '''
    cv2.imshow("out_Video", IMG_OUT)  # 显示
    cv2.waitKey(1000 // int(fps))  # 延迟
    '''
    videoWriter.write(IMG_OUT)

    success, frame = videoCapture.read()
    count += 1
    print(count)
print('end')


