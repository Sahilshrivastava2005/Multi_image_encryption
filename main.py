from modules import image_utils
from modules import encryption
from modules import decryption
import config


def main():

    print("FINAL TEST – Encryption + Decryption")
    # Test indexed transform reversibility

    img1 = config.INPUT_PATH + "img1.png"
    img2 = config.INPUT_PATH + "img2.png"
    img3 = config.INPUT_PATH + "img3.png"

    I1, I2, I3 = image_utils.prepare_images(img1, img2, img3)

    (I1_index, MAP1), (I2_index, MAP2), (I3_index, MAP3) = \
        image_utils.prepare_indexed_images(I1, I2, I3)

    user_key = "MY_SECRET_KEY_123"
    
    rgb_test = image_utils.indexed_to_rgb(I1_index, MAP1)

    image_utils.save_image(rgb_test, "indexed_reconstruction.png")
    # ---------- ENCRYPTION ----------

    (S1, k1), (S2, k2), (S3, k3) = encryption.scramble_all_images(
        I1_index, I2_index, I3_index, user_key
    )

    D1, D2, D3 = encryption.diffuse_all_images(
        S1, S2, S3, user_key
    )

    # ---------- DECRYPTION ----------

    RS1, RS2, RS3 = decryption.reverse_diffusion_all(
        D1, D2, D3, user_key
    )

    RI1, RI2, RI3 = decryption.reverse_scramble_all(
        RS1, RS2, RS3, k1, k2, k3
    )
    # Test scrambling reversibility

    rev = decryption.reverse_scramble(S1, k1)

    image_utils.save_image(
        image_utils.indexed_to_rgb(rev, MAP1),
        "scramble_reverse_test.png"
    )
    
    test = decryption.reverse_diffusion(D1, user_key)

    image_utils.save_image(
        image_utils.indexed_to_rgb(test, MAP1),
        "diffusion_reverse_test.png"
    )

    # ---------- SAVE RESULTS ----------

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

    print("\nDecryption Completed Successfully!")
    print("Check images/output folder for results")


if __name__ == "__main__":
    main()
