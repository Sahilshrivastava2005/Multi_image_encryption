from PIL import Image
import numpy as np
import config
import cv2
import os


# ==========================================================
# BASIC IMAGE LOADING / SAVING
# ==========================================================

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Unable to read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_image(img):
    return cv2.resize(img, config.IMAGE_SIZE)


def prepare_images(img1_path, img2_path, img3_path):

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

    path = os.path.join(config.OUTPUT_PATH, filename)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

    print(f"[INFO] Image saved at: {path}")


# ==========================================================
# INDEXED IMAGE CONVERSION (USED BY ENCRYPTION)
# ==========================================================

def indexed_image_conversion(img):
    """
    Convert RGB image to indexed format (256 colors).
    Returns:
        index_matrix (2D uint8)
        palette (768-length list)
    """

    pil_img = Image.fromarray(img)

    # Convert to 8-bit indexed image
    indexed = pil_img.quantize(
        colors=config.INDEXED_COLORS,
        method=Image.MEDIANCUT
    )

    index_matrix = np.array(indexed, dtype=np.uint8)
    palette = indexed.getpalette()

    return index_matrix, palette


def indexed_to_rgb(index_matrix, palette):
    """
    Convert indexed image back to RGB using palette.
    """

    indexed_img = Image.fromarray(index_matrix.astype(np.uint8), mode='P')
    indexed_img.putpalette(palette)

    rgb_img = indexed_img.convert("RGB")

    return np.array(rgb_img)


# ==========================================================
# BULK INDEXED PREPARATION
# ==========================================================

def prepare_indexed_images(I1, I2, I3):

    print("[INFO] Converting images to indexed format...")

    I1_index, MAP1 = indexed_image_conversion(I1)
    I2_index, MAP2 = indexed_image_conversion(I2)
    I3_index, MAP3 = indexed_image_conversion(I3)

    print("[INFO] Indexed image conversion completed")

    return (I1_index, MAP1), (I2_index, MAP2), (I3_index, MAP3)


# ==========================================================
# RGB RECOVERY AFTER DECRYPTION
# ==========================================================

def recover_rgb_images(CR, CG, CB, MAP1, MAP2, MAP3):
    """
    Convert decrypted indexed channels back to RGB images.
    """

    R_img = indexed_to_rgb(CR, MAP1)
    G_img = indexed_to_rgb(CG, MAP2)
    B_img = indexed_to_rgb(CB, MAP3)

    return R_img, G_img, B_img

# ==========================================================
# INDEXED IMAGE CONVERSION (USED BY ENCRYPTION)
# ==========================================================

def indexed_image_conversion(img):
    """
    Convert RGB image to indexed format (256 colors).
    Returns:
        index_matrix (2D uint8)
        palette (768-length list)
    """
    # ... (Your existing code)
    pil_img = Image.fromarray(img)
    
    # Assuming config.INDEXED_COLORS = 256
    indexed = pil_img.quantize(
        colors=256, 
        method=Image.MEDIANCUT
    )

    index_matrix = np.array(indexed, dtype=np.uint8)
    palette = indexed.getpalette()

    return index_matrix, palette


def inverse_indexed_image_conversion(index_matrix, palette):
    """
    Convert indexed image back to RGB using palette.
    (Renamed to match the decryption module call)
    """
    indexed_img = Image.fromarray(index_matrix.astype(np.uint8), mode='P')
    indexed_img.putpalette(palette)

    rgb_img = indexed_img.convert("RGB")

    return np.array(rgb_img)

# Alias for your internal bulk recovery function
