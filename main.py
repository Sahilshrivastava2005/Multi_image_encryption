from modules import image_utils
from modules import encryption
from modules import decryption
import config
import numpy as np


def main():

    print("\n===== FINAL TEST – Encryption + Decryption =====\n")

    img1 = config.INPUT_PATH + "img1.png"
    img2 = config.INPUT_PATH + "img2.png"
    img3 = config.INPUT_PATH + "img3.png"

    # -------------------------------------------------
    # Load RGB images
    # -------------------------------------------------
    P1_rgb, P2_rgb, P3_rgb = image_utils.prepare_images(
        img1, img2, img3
    )

    # -------------------------------------------------
    # Prepare indexed images + store palettes
    # -------------------------------------------------
    (I1_index, MAP1), (I2_index, MAP2), (I3_index, MAP3) = \
        image_utils.prepare_indexed_images(P1_rgb, P2_rgb, P3_rgb)

    # -------------------------------------------------
    # USER SECRET KEY
    # -------------------------------------------------
    user_key = input("Enter secret key: ").strip()

    if len(user_key) < 8:
        raise ValueError("Key must be at least 8 characters.")

    # -------------------------------------------------
    # ENCRYPTION
    # -------------------------------------------------
    cipher_image = encryption.encrypt_three_images(
        P1_rgb,
        P2_rgb,
        P3_rgb,
        user_key
    )

    image_utils.save_image(cipher_image, "final_cipher.png")
    print("[SUCCESS] Encryption completed.")

    # -------------------------------------------------
    # DECRYPTION (returns RGB images directly)
    # -------------------------------------------------
    P1_dec, P2_dec, P3_dec = decryption.decrypt_three_images(
        cipher_image,
        user_key,
        MAP1, MAP2, MAP3
    )

    image_utils.save_image(P1_dec, "dec1.png")
    image_utils.save_image(P2_dec, "dec2.png")
    image_utils.save_image(P3_dec, "dec3.png")

    print("[SUCCESS] Decryption completed.")

    # -------------------------------------------------
    # Verification Check
    # -------------------------------------------------
    if (
        np.array_equal(P1_rgb, P1_dec) and
        np.array_equal(P2_rgb, P2_dec) and
        np.array_equal(P3_rgb, P3_dec)
    ):
        print("✅ PERFECT REVERSIBILITY CONFIRMED")
    else:
        print("❌ ERROR: Decryption mismatch")


if __name__ == "__main__":
    main()
