from PIL import Image
import numpy as np
import config
import cv2
import os


def load_image(path):
    """
    Load an image from disk in RGB format
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Unable to read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_image(img):
    """
    Resize image to standard dimensions defined in config
    """
    return cv2.resize(img, config.IMAGE_SIZE)


def prepare_images(img1_path, img2_path, img3_path):
    """
    Load and preprocess three images for encryption
    """

    print("[INFO] Loading images...")

    I1 = load_image(img1_path)
    I2 = load_image(img2_path)
    I3 = load_image(img3_path)

    print("[INFO] Resizing images to:", config.IMAGE_SIZE)

    I1 = resize_image(I1)
    I2 = resize_image(I2)
    I3 = resize_image(I3)

    print("[INFO] Images successfully prepared")

    return I1, I2, I3


def save_image(img, filename):
    """
    Save image to output directory
    """
    path = os.path.join(config.OUTPUT_PATH, filename)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

    print(f"[INFO] Image saved at: {path}")



def rgb_to_indexed(img):
    """
    Convert RGB image to indexed image with palette
    Returns:
        index_matrix: 2D array of indices
        palette: color palette map
    """

    pil_img = Image.fromarray(img)

    # Convert to indexed image (256 colors)
    indexed = pil_img.quantize(colors=config.INDEXED_COLORS)

    # Get palette map
    palette = indexed.getpalette()

    # Convert indexed image to numpy array
    index_matrix = np.array(indexed)

    return index_matrix, palette


def indexed_to_rgb(index_matrix, palette):
    """
    Convert indexed image back to RGB using palette map
    """

    h, w = index_matrix.shape

    rgb_image = Image.fromarray(index_matrix.astype(np.uint8), mode='P')
    rgb_image.putpalette(palette)

    rgb_image = rgb_image.convert("RGB")

    return np.array(rgb_image)


def prepare_indexed_images(I1, I2, I3):
    """
    Convert three RGB images to indexed format
    """

    print("[INFO] Converting images to indexed format...")

    I1_index, MAP1 = rgb_to_indexed(I1)
    I2_index, MAP2 = rgb_to_indexed(I2)
    I3_index, MAP3 = rgb_to_indexed(I3)

    print("[INFO] Indexed image conversion completed")

    return (I1_index, MAP1), (I2_index, MAP2), (I3_index, MAP3)
