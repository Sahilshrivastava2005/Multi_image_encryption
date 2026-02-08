from modules import image_utils
from modules import encryption
from modules import decryption
import config


def main():

    print("\n===== FINAL TEST – Encryption + Decryption =====\n")

    img1 = config.INPUT_PATH + "img1.png"
    img2 = config.INPUT_PATH + "img2.png"
    img3 = config.INPUT_PATH + "img3.png"

    I1, I2, I3 = image_utils.prepare_images(img1, img2, img3)

    (I1_index, MAP1), (I2_index, MAP2), (I3_index, MAP3) = \
        image_utils.prepare_indexed_images(I1, I2, I3)

    user_key = "MY_SECRET_KEY_123"

    # ---------- ENCRYPTION ----------

    (S1, k1), (S2, k2), (S3, k3) = encryption.scramble_all_images(
        I1_index, I2_index, I3_index, user_key
    )

    D1, D2, D3 = encryption.diffuse_all_images(
        S1, S2, S3, user_key
    )

    image_utils.save_image(D1, "cipher1.png")
    image_utils.save_image(D2, "cipher2.png")
    image_utils.save_image(D3, "cipher3.png")

    # ---------- DECRYPTION ----------

    RS1, RS2, RS3 = decryption.reverse_diffusion_all(
        D1, D2, D3, user_key
    )

    RI1, RI2, RI3 = decryption.reverse_scramble_all(
        RS1, RS2, RS3, k1, k2, k3
    )

    # ---------- VALIDATION ----------

    print("\n===== VALIDATION =====")

    print("Image1 match:", (I1_index == RI1).all())
    print("Image2 match:", (I2_index == RI2).all())
    print("Image3 match:", (I3_index == RI3).all())

    # ---------- SAVE DECRYPTED ----------

    image_utils.save_image(
        image_utils.indexed_to_rgb(RI1, MAP1),
        "decrypted1.png"
    )

    image_utils.save_image(
        image_utils.indexed_to_rgb(RI2, MAP2),
        "decrypted2.png"
    )

    image_utils.save_image(
        image_utils.indexed_to_rgb(RI3, MAP3),
        "decrypted3.png"
    )

    print("\nDecryption Completed Successfully!\n")


if __name__ == "__main__":
    main()
