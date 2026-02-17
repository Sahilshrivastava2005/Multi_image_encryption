import numpy as np
import config
from modules import image_utils, encryption, decryption

def main():
    print("\n===== FINAL TEST – Encryption + Decryption =====\n")

    img1 = config.INPUT_PATH + "img1.png"
    img2 = config.INPUT_PATH + "img2.png"
    img3 = config.INPUT_PATH + "img3.png"

    # 1. Load RGB images
    P1_rgb, P2_rgb, P3_rgb = image_utils.prepare_images(img1, img2, img3)

    # 2. Generate Indexed Images and Palettes ONCE
    (I1_index, MAP1), (I2_index, MAP2), (I3_index, MAP3) = \
        image_utils.prepare_indexed_images(P1_rgb, P2_rgb, P3_rgb)

    user_key = "sjakkakaka"

    # 3. ENCRYPTION - Pass the INDEXED images, not the RGB ones!
    cipher_image = encryption.encrypt_three_images(
        I1_index, I2_index, I3_index, user_key
    )
    
    image_utils.save_image(cipher_image, "final_cipher.png")
    print("[SUCCESS] Encryption completed.")

    # 4. DECRYPTION
    P1_dec, P2_dec, P3_dec = decryption.decrypt_three_images(
        cipher_image, user_key, MAP1, MAP2, MAP3
    )
    print("Original R sum:", np.sum(P1_rgb[:, :, 0]))
    print("Decrypted R sum:", np.sum(P1_dec[:, :, 0]))

    image_utils.save_image(P1_dec, "dec1.png")
    image_utils.save_image(P2_dec, "dec2.png")
    image_utils.save_image(P3_dec, "dec3.png")
    print("[SUCCESS] Decryption completed.")

    # 5. THE TRUTH TEST - Compare against the quantized versions
    P1_quantized = image_utils.indexed_to_rgb(I1_index, MAP1)
    P2_quantized = image_utils.indexed_to_rgb(I2_index, MAP2)
    P3_quantized = image_utils.indexed_to_rgb(I3_index, MAP3)

    if (np.array_equal(P1_quantized, P1_dec) and
        np.array_equal(P2_quantized, P2_dec) and
        np.array_equal(P3_quantized, P3_dec)):
        print("✅ PERFECT REVERSIBILITY CONFIRMED")
    else:
        print("❌ ERROR: Decryption mismatch")
        # === DIAGNOSTIC TEST ===
        # Check if the generated palettes somehow drifted
        print(f"Palette 1 Check: {np.array_equal(P1_quantized, P1_dec)}")
        print(f"Palette 2 Check: {np.array_equal(P2_quantized, P2_dec)}")
        print(f"Palette 3 Check: {np.array_equal(P3_quantized, P3_dec)}")
        
        # Check if the chaos module is stateful (yielding different sequences)
        D_enc = encryption.generate_chaos_sequences(user_key, I1_index.shape[0] * I1_index.shape[1])
        D_dec = decryption.generate_chaos_sequences(user_key, I1_index.shape[0] * I1_index.shape[1])
        print(f"Chaos Generator is deterministic: {np.array_equal(D_enc[0], D_dec[0])}")
        
        # 5. THE TRUTH TEST
        print("\n--- MATRIX PURITY CHECK ---")
        
        # 5a. Test the RAW math logic BEFORE color map conversion
        # We decrypt the indices manually to see if the math succeeded
        H, W = cipher_image.shape[0], cipher_image.shape[1]
        Keys = decryption.derive_key_parts(user_key)
        _, IC1, IC2, Fmat = decryption.build_fractal_matrix(H, W, Keys)
        D = decryption.generate_chaos_sequences(user_key, H * W)
        I1_rec, I2_rec, I3_rec = decryption.synchronized_disorder_decryption(
            cipher_image, IC1, IC2, Fmat, D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7]
        )
        
        # Cast your original index directly to uint8 exactly like encryption did
        I1_original_uint8 = I1_index.astype(np.uint8)
        
        if np.array_equal(I1_original_uint8, I1_rec):
            print("✅ PURE MATHEMATICAL DECRYPTION: SUCCESS (Arrays Match 100%)")
        else:
            print("❌ PURE MATHEMATICAL DECRYPTION: FAILED")
            
        # 5b. Test the final RGB Mapping logic
        P1_quantized = image_utils.indexed_to_rgb(I1_index, MAP1)
        if np.array_equal(P1_quantized, P1_dec) and np.array_equal(P2_quantized, P2_dec) and np.array_equal(P3_quantized, P3_dec):
            print("✅ PERFECT REVERSIBILITY CONFIRMED")
        else:
            print("❌ ERROR: Decryption mismatch in RGB Mapping phase")
if __name__ == "__main__":
    main()