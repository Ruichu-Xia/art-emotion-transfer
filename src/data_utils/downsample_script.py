import os
from PIL import Image


def downsample_images(input_dir, output_dir, target_size=(224, 224)):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        # Create corresponding output directory
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subdir, file)
                
                try:
                    # Open and resize the image
                    with Image.open(input_path) as img:
                        img_resized = img.resize(target_size, Image.LANCZOS)
                        img_resized.save(output_path)
                        print(f"Processed: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    input_directory = "data/wikiart"
    output_directory = "data/wikiart_resized"
    downsample_images(input_directory, output_directory)

