import numpy as np
import cv2
import math


# -------- ENTROPY --------

def entropy(image):
    """
    Calculate Shannon entropy of an image
    """

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()

    ent = 0
    for p in hist:
        if p > 0:
            ent -= p * math.log2(p)

    return float(ent)


# -------- HISTOGRAM ANALYSIS --------

def histogram(image):
    return cv2.calcHist([image], [0], None, [256], [0, 256])


# -------- NPCR --------

def npcr(img1, img2):
    """
    Number of Pixel Change Rate
    """

    h, w = img1.shape

    diff = img1 != img2

    change = np.sum(diff)

    return (change / (h * w)) * 100


# -------- UACI --------

def uaci(img1, img2):
    """
    Unified Average Changing Intensity
    """

    diff = np.abs(img1.astype(int) - img2.astype(int))

    return np.mean(diff) / 255 * 100


# -------- CORRELATION --------

def correlation(image):
    """
    Adjacent pixel correlation
    """

    img = image.flatten()

    x = img[:-1]
    y = img[1:]

    corr = np.corrcoef(x, y)[0, 1]

    return corr


# -------- KEY SENSITIVITY --------

def key_sensitivity_test(encrypt_function, img, key1, key2):
    """
    Encrypt same image with two slightly different keys
    and compute NPCR between results
    """

    c1 = encrypt_function(img, key1)
    c2 = encrypt_function(img, key2)

    return npcr(c1, c2)


# -------- COMPLETE REPORT --------

def security_report(original, cipher):
    """
    Generate full security analysis report
    """

    report = {}

    report["Entropy"] = entropy(cipher)
    report["Correlation"] = correlation(cipher)

    return report
