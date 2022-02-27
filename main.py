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
            cur = cv.resize(cur, (cur.shape[1]//2, cur.shape[0]//2), interpolation = cv.INTER_AREA)
            cur[cur == 71] = 0
            cur[cur == 70] = 0
            if cur is not None:
                images.append(np.expand_dims(np.transpose(cur, (1,0)), axis=0))
    return images


def show(images : np.ndarray):
    """
    Make grid and show images, 3 img in column

    Parameters:
    ----------
    images -> np.ndarray
    """
    x = torch.from_numpy(images)
    grid = torchvision.utils.make_grid(x, 3)
    plt.imshow(np.transpose(grid, (2,1,0)))
    plt.show()


def get_masks(img0 : np.ndarray, draw = False) -> np.ndarray:
    """
    Thish fuction find closed contours, fill them and make masks

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
    contours, _ = cv.findContours(gray, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)
    for i in contours:
        if cv.contourArea(i) > 5000 and cv.contourArea(i) < 60000:
            cnt+=1
            m = np.zeros(img0[0].shape)
            cv.fillPoly(m, pts = [i], color = (255,255,0))
            mask.append(m)
            cv.drawContours(img0[0], [i], 0, (255,255,0), 5)
    if draw:
        cv.imshow("img", np.transpose(img0[0], (1,0)))
        if cv.waitKey(0) & 0xFF == ord('q'):
            cv.destroyAllWindows()
    return np.array(mask)


def proector(masks : np.ndarray, img1 : np.ndarray):
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


def plausibility(img0 : np.ndarray, img1 : np.ndarray, alpha = 0.1):
    """
    This fuction defines sameness objects on img0 and img1

    Parameters:
    -----------
    img0: np.ndarray
    img1: np.ndarray
    alpha: np.float8 - maximum deviation from the img0
    """
    proection = proector(get_masks(img0), img1[0])
    if np.sum(np.abs(img1 - proection)) < alpha * np.sum(img1):
        return True
    else: return False


def show_masks(images : np.ndarray):
    mask = []
    for i in range(len(images)):
        mask.append(get_masks(images[i]))
    show(np.array(mask))


if __name__ == "__main__":
    data_kube = np.array(load_images("data/kube"))
    data_ball = np.array(load_images("data/ball"))
    show(data_kube)
    show_masks(data_kube)
    if plausibility(data_kube[4], data_kube[8]): print("The objects are the same")
    else: print("the objects are NOT the same")