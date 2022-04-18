import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys

np.set_printoptions(threshold=sys.maxsize)


def load_images(folder):
    """
    Load imgs in np.ndarray (B, C, H, W)
    """
    images = []
    for _ in sorted(os.listdir(folder)):
        if _ != ".DS_Store":
            cur = cv.imread(os.path.join(folder, _))
            cur = cv.cvtColor(cur, cv.COLOR_BGR2GRAY)
            cur = cv.resize(cur, (cur.shape[1] // 2, cur.shape[0] // 2),
                            interpolation=cv.INTER_AREA)
            cur[cur == 71] = 0
            cur[cur == 70] = 0
            # print(cur.shape)
            if cur is not None:
                images.append(np.expand_dims(np.transpose(cur, (1, 0)),
                                             axis=0))
    return images


def show(images: np.ndarray):
    """
    Make grid and show images, 3 img in column

    Parameters:
    ----------
    images -> np.ndarray
    """
    x = torch.from_numpy(images)
    grid = torchvision.utils.make_grid(x, 3)
    plt.imshow(np.transpose(grid, (2, 1, 0)))
    plt.show()


def get_masks(img0: np.ndarray, draw=False) -> np.ndarray:
    """
    This fuction find closed contours, fill them and make masks

    Parameters:
    -----------
    img0: np.ndarray
    draw: boolean
    """
    cnt = 0
    mask = []
    gray = cv.Canny(img0[0], 5, 5)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(gray, cv.RETR_LIST,
                                  cv.CHAIN_APPROX_TC89_KCOS)
    for i in contours:
        if cv.contourArea(i) > 5000 and cv.contourArea(i) < 60000:
            cnt += 1
            m = np.zeros(img0[0].shape)
            cv.fillPoly(m, pts=[i], color=(255, 255, 0))
            mask.append(m)
            cv.drawContours(img0[0], [i], 0, (255, 255, 0), 5)
    if draw:
        cv.imshow("img", np.transpose(img0[0], (1, 0)))
        if cv.waitKey(0) & 0xFF == ord('q'):
            cv.destroyAllWindows()
    return np.array(mask)


def proector(masks: np.ndarray, img1: np.ndarray):
    """
    Get projection img1 on img0

    Parameters:
    -----------
    mask : np.ndarray
    img1 : np.ndarray
    """
    img = np.copy(img1)
    for i in masks:
        img[i > 0] = np.sum(img[i > 0]) / np.sum(i > 0)
    return img


def plausibility(img0_: np.ndarray, img1_: np.ndarray, alpha=0.3):
    """
    This fuction defines sameness objects on img0 and img1

    Parameters:
    -----------
    img0: np.ndarray
    img1: np.ndarray
    alpha: np.float8 - maximum deviation from the img0
    """
    proection = proector(get_masks(img0_), img1_[0])
    a = np.sum(img1_ - proection)
    b = alpha * np.sum(img1_)
    print(np.sum(img1_ - img0_), a)
    if a < b:
        return True
    else:
        return False


def show_masks(img: np.ndarray):
    """
    Just show masks of all images

    Parameters:
    -----------
    images: np.ndarray
    """
    images = np.copy(img)
    mask = []
    for i in range(len(images)):
        mask.append(get_masks(images[i]))
    show(np.array(mask))


def generalCase(data: np.ndarray):
    matrix = data.reshape((len(data), -1))
    eig_val, eig_vect = np.linalg.eig(matrix.dot(matrix.T))
    max_eigVect = eig_vect[:, np.argmax(eig_val)].reshape((-1, 1))
    return matrix.T.dot(max_eigVect)


def comparison(vec1: np.ndarray, vec2: np.ndarray, img: np.ndarray):
    """
    This fuction defines the type of object in the image (ball || square) in the general case
    
    Parameters:
    -----------
    vec1 : np.ndarray - main vector of kube images
    vec2 : np.ndarray - main vector of ball images
    img : np.ndarray - unknown image

    Returns:
    --------
    type of unknown img
    """
    sos = img.reshape((1, -1))
    if np.linalg.norm(vec1 - sos) < np.linalg.norm(vec2 - sos):
        return "квадрат"
    else:
        return "круг"


if __name__ == "__main__":
    data_kube = np.array(load_images("data/kube"))
    data_ball = np.array(load_images("data/ball"))
    ball_vector = generalCase(data_ball[:4])
    kube_vector = generalCase(data_kube[4:])
    comparison(kube_vector.reshape((1, -1)), ball_vector.reshape((1, -1)),
                data_kube[3])
    show(data_kube)
    show(data_ball)
    get_masks(data_ball[2], True)
    get_masks(data_kube[2], True)
    show_masks(data_kube)
#    if plausibility(data_kube[0], data_ball[0]): print("The objects are the same")
#    else: print("the objects are NOT the same")
