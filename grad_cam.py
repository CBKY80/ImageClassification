import tensorflow as tf
import os
import argparse
import numpy as np
import cv2

# Suppress informational messages from TensorFlow
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# --- CORE GRAD-CAM & HELPER FUNCTIONS ---

def get_img_array(img_path, size):
    """
    Loads an image from a file path and converts it to a NumPy array.
    """
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    return array

def make_gradcam_heatmap(processed_img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates the Grad-CAM heatmap from a preprocessed image array.
    """
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(processed_img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam_visualization(img_path, heatmap, cam_path="gradcam_visualization.jpg", alpha=0.4):
    """
    Saves the Grad-CAM visualization by overlaying the heatmap on the image.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cv2.COLORMAP_JET
    jet_heatmap = cv2.applyColorMap(heatmap_uint8, jet)
    jet_heatmap_resized = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    superimposed_img = jet_heatmap_resized * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    cv2.imwrite(cam_path, superimposed_img)
    print(f"\n--- Grad-CAM visualization successfully saved to '{cam_path}' ---")

# --- UNIFIED CLASSIFICATION FUNCTION ---
def classify_and_print(model, img_array, header_text):
    """
    Takes a raw image array (0-255), preprocesses it, predicts, and prints results.
    """
    print(f"\n--- {header_text} ---")
    
    # Add batch dimension and preprocess for the model
    img_batch = np.expand_dims(img_array, axis=0)
    processed_batch = preprocess_input(img_batch)
    
    preds = model.predict(processed_batch)
    
    print("Top-3 Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decode_predictions(preds, top=3)[0]):
        print(f"{i + 1}: {label} ({score:.2f})")
    return preds

# --- OCCLUSION FUNCTIONS (SAVE THE FILE AND RETURN THE IMAGE ARRAY) ---

def create_mask_from_heatmap(heatmap, threshold_value=0.6):
    """
    Creates a binary mask from a heatmap by applying a threshold.
    """
    threshold = np.max(heatmap) * threshold_value
    mask = (heatmap > threshold).astype(np.uint8)
    return mask

def occlude_with_mean_color(image, mask):
    """
    Occludes with mean color, saves the file, and returns the array.
    """
    mean_color = cv2.mean(image)[:3]
    occluded_image = image.copy()
    for c in range(3):
        occluded_image[:, :, c] = np.where(mask == 1, mean_color[c], occluded_image[:, :, c])
    cv2.imwrite("occluded_mean.jpg", occluded_image)
    print("--- Mean color occlusion image saved to 'occluded_mean.jpg' ---")
    return occluded_image

def occlude_with_blur(image, mask, kernel_size=(99, 99)):
    """
    Occludes with blur, saves the file, and returns the array.
    """
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    mask_3d = np.stack([mask] * 3, axis=-1)
    occluded_image = np.where(mask_3d == 1, blurred_image, image)
    cv2.imwrite("occluded_blur.jpg", occluded_image)
    print("--- Blur occlusion image saved to 'occluded_blur.jpg' ---")
    return occluded_image

def occlude_with_pixel_scramble(image, mask):
    """
    Occludes with pixel scramble, saves the file, and returns the array.
    """
    occluded_image = image.copy()
    region_indices = np.where(mask == 1)
    region_pixels = image[region_indices]
    np.random.shuffle(region_pixels)
    occluded_image[region_indices] = region_pixels
    cv2.imwrite("occluded_scramble.jpg", occluded_image)
    print("--- Pixel scramble occlusion image saved to 'occluded_scramble.jpg' ---")
    return occluded_image


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate Grad-CAM and run predictions on occluded images.")
    parser.add_argument("--occlude", type=str, choices=['mean', 'blur', 'scramble', 'all'],
                        help="Type of occlusion to apply for re-prediction. 'all' tests every method.")
    args = parser.parse_args()

    # --- Configuration ---
    IMAGE_PATH = 'basic_cat.jpg'
    LAST_CONV_LAYER_NAME = 'out_relu'
    
    if not os.path.exists(IMAGE_PATH):
        print(f"FATAL ERROR: Image file not found at '{IMAGE_PATH}'")
        exit()
        
    print("--- Loading model... ---")
    model = MobileNetV2(weights="imagenet")
    model.layers[-1].activation = tf.keras.activations.linear

    # --- Step 1: Prediction on ORIGINAL image to generate the heatmap ---
    original_img_array = get_img_array(IMAGE_PATH, size=(224, 224))
    original_preds = classify_and_print(model, original_img_array.copy(), "Predictions on ORIGINAL Image")

    # --- Step 2: Generate and save the Grad-CAM visualization ---
    # We need to manually preprocess the image array again for the make_gradcam_heatmap function
    processed_original_array = preprocess_input(np.expand_dims(original_img_array.copy(), axis=0))
    heatmap = make_gradcam_heatmap(processed_original_array, model, LAST_CONV_LAYER_NAME, pred_index=np.argmax(original_preds[0]))
    save_gradcam_visualization(IMAGE_PATH, heatmap)
    
    # --- Step 3: Occlude the image and re-predict if requested ---
    if args.occlude:
        # Resize heatmap to image size to create a correctly scaled mask
        heatmap_resized_for_mask = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]))
        mask = create_mask_from_heatmap(heatmap_resized_for_mask)

        occlusion_methods = {
            'mean': occlude_with_mean_color,
            'blur': occlude_with_blur,
            'scramble': occlude_with_pixel_scramble
        }
        
        # Determine which occlusion methods to run based on the '--occlude' argument
        methods_to_run = occlusion_methods.keys() if args.occlude == 'all' else [args.occlude]
        
        for method_name in methods_to_run:
            # Get the occluded image in memory from the corresponding function
            occluded_image_array = occlusion_methods[method_name](original_img_array.copy(), mask)
            # Run the prediction on the newly created occluded image
            classify_and_print(model, occluded_image_array, f"Predictions for '{method_name.upper()}' Occlusion")

    print("\n--- Script finished successfully. ---")