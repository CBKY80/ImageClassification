from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import argparse
import os

def apply_selected_filter(image_path, filter_name):
    """
    Applies a chosen filter to an image and saves the result.
    """
    # --- 1. Define the available filter operations ---
    # The dictionary now holds lambda functions, making it flexible enough
    # to handle both simple filters and custom multi-step sequences like 'halftone'.
    available_filters = {
        'blur': lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)),
        'sharpen': lambda img: img.filter(ImageFilter.SHARPEN),
        'edges': lambda img: img.filter(ImageFilter.FIND_EDGES),
        'emboss': lambda img: img.filter(ImageFilter.EMBOSS),
        'halftone': lambda img: img.convert('L').convert('1', dither=Image.Dither.FLOYDSTEINBERG)
    }

    try:
        # Determine which filters to run based on the 'filter_name' argument
        if filter_name == 'all':
            filters_to_run = available_filters.items()
        else:
            # If not 'all', create a list containing just the one selected filter
            filters_to_run = [(filter_name, available_filters[filter_name])]

        # --- 2. Load and resize the base image once ---
        print(f"Loading image from '{image_path}'...")
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))

        # --- 3. Loop through the selected filter(s), apply, and save ---
        for name, filter_function in filters_to_run:
            print(f"Applying '{name}' filter...")
            
            # Apply the filter function to the resized image
            img_filtered = filter_function(img_resized)
            
            # Prepare to save the image
            output_filename = f"filtered_{name}.png"
            # Use a grayscale colormap for filters that produce non-RGB images (like edges/halftone)
            cmap = 'gray' if img_filtered.mode != 'RGB' else None
            plt.imshow(img_filtered, cmap=cmap)
            plt.axis('off')
            plt.savefig(output_filename)
            
            print(f"  -> Saved result to '{output_filename}'")

        print("\nProcessing complete.")

    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except KeyError:
        print(f"Error: Filter '{filter_name}' is not a valid choice.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- Set up command-line argument parsing to accept user input ---
    parser = argparse.ArgumentParser(
        description="Apply a specified image filter to the 'basic_cat.jpg' image."
    )
    parser.add_argument(
        "-f", "--filter",
        type=str,
        required=True,
        # The list of choices is updated with the new 'halftone' filter
        choices=['blur', 'sharpen', 'edges', 'emboss', 'halftone', 'all'],
        help="The filter to apply. Use 'all' to apply and save every filter."
    )
    args = parser.parse_args()

    # Define the input image
    input_image_path = "basic_cat.jpg"
    
    # Call the main function with the user's chosen filter
    apply_selected_filter(input_image_path, args.filter)