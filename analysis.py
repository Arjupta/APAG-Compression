import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import apag_compression as apag


#Calculate RMSE between the original and compressed images.

def calculate_rmse(original, compressed):
    original_array = np.array(original, dtype=np.float32)
    compressed_array = np.array(compressed, dtype=np.float32)
    mse = np.mean((original_array - compressed_array) ** 2)
    rmse = np.sqrt(mse)
    return rmse


# Calculate Bits Per Pixel (BPP) given the file size in bytes and image dimensions

def calculate_bpp(file_size, width, height):
    total_pixels = width * height
    bpp = (file_size * 8) / total_pixels
    return bpp


# Standard jpeg compression 

def jpeg_compression_metrics(original_image_path, filename):
    # Load the original image
    original_image = Image.open(original_image_path) #.convert("RGB")
    width, height = original_image.size

    # Define JPEG quality levels to test
    quality_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    rmse_values = []
    bpp_values = []

    for quality in quality_levels:
        # Save the image as JPEG with the specified quality
        compressed_image_name = f"{filename}_quality_{quality}"
        compressed_image_path = "./output/jpeg/" + compressed_image_name + ".jpeg"
        original_image.save(compressed_image_path, "JPEG", quality=quality)

        # Reload the compressed image
        compressed_image = Image.open(compressed_image_path) #.convert("RGB")

        # Calculate RMSE
        rmse = calculate_rmse(original_image, compressed_image)
        rmse_values.append(rmse)

        # Calculate BPP
        file_size = os.path.getsize(compressed_image_path)  # File size in bytes
        bpp = calculate_bpp(file_size, width, height)
        bpp_values.append(bpp)

        # Clean up compressed file if needed
        #os.remove(compressed_image_path)

    return (rmse_values, bpp_values)

def apag_compression_metrics(original_image_path, filename):
    # Load the original image
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = original_image.shape
    new_h = (height + 7) // 8 * 8
    new_w = (width + 7) // 8 * 8
    padded_image = np.full((new_h, new_w), 255, dtype=np.uint8)
    padded_image[:height, :width] = original_image
    original_image = padded_image
    
    # Define JPEG quality levels to test
    quality_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    rmse_values = []
    bpp_values = []

    for quality in quality_levels:
        # Save the image as JPEG with the specified quality
        compressed_image_name = f"{filename}_quality_{quality}"
        compressed_image_path = "./output/apag/" + compressed_image_name + ".apag"

        apag.generate_compressed_file(original_image, compressed_image_path, quality)

        # Reload the compressed image
        compressed_image = apag.decompress_image_from_file(compressed_image_path, compressed_image_name)

        # Calculate RMSE
        rmse = calculate_rmse(original_image, compressed_image)
        rmse_values.append(rmse)

        # Calculate BPP
        file_size = os.path.getsize(compressed_image_path)  # File size in bytes
        bpp = calculate_bpp(file_size, width, height)
        bpp_values.append(bpp)

        # Clean up compressed file if needed
        #os.remove(compressed_image_path)

    return (rmse_values, bpp_values)


# Plot BPP vs RMSE

def plot_bpp_vs_rmse(jpeg_data, apag_data, name):
    plt.figure(figsize=(8, 6))

    rmse_values, bpp_values = jpeg_data
    plt.plot(bpp_values, rmse_values, marker="o", color="blue", label="JPEG")

    rmse_values, bpp_values = apag_data
    plt.plot(bpp_values, rmse_values, marker="x", color="red", label="APAG")

    plt.title("BPP vs RMSE Comparison", fontsize=14)
    plt.xlabel("Bits Per Pixel (BPP)", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.grid()
    plt.legend()
    plt.savefig(f"./output/plots/{name}_plot.png", dpi=300, format="png")
    #plt.show()
    

# Plots all APAG RMSE vs BPP curves in a single plot.

def plot_all_apag_compressions(apag_data_list, image_names):
    plt.figure(figsize=(10, 8))

    # Loop over each APAG data set and corresponding image name
    for apag_data, name in zip(apag_data_list, image_names):
        rmse_values, bpp_values = apag_data
        plt.plot(bpp_values, rmse_values, marker="x", label=f"APAG - {name}")

    # Configure plot settings
    plt.title("BPP vs RMSE for All Images (APAG Compression)", fontsize=14)
    plt.xlabel("Bits Per Pixel (BPP)", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.grid()
    plt.legend()
    plt.savefig("./output/plots/apag_all_images_plot.png", dpi=300, format="png")
    #plt.show()

def cleanup():
    directory_jpeg = "./output/jpeg"
    directory_apag = "./output/apag"
    directory_plot = "./output/plots"

    for filename in os.listdir(directory_jpeg):
        file_path = os.path.join(directory_jpeg, filename)
        os.unlink(file_path)
    for filename in os.listdir(directory_apag):
        file_path = os.path.join(directory_apag, filename)
        os.unlink(file_path)
    for filename in os.listdir(directory_plot):
        file_path = os.path.join(directory_plot, filename)
        os.unlink(file_path)

if __name__ == "__main__":
    cleanup()
    test_directory = './test_imgs'
    all_apag_data = []  # Store APAG data for all images
    image_names = []    # Store image names

    for filename in os.listdir(test_directory):
        name, ext = os.path.splitext(filename)
        original_image_path = test_directory + '/' + filename
    
        jpeg_data = jpeg_compression_metrics(original_image_path, name)
        apag_data = apag_compression_metrics(original_image_path, name)

        # Plot the BPP vs RMSE curve
        plot_bpp_vs_rmse(jpeg_data, apag_data, name)

        all_apag_data.append(apag_data)
        image_names.append(name)

    # Plot all APAG RMSE vs BPP curves in a single plot
    plot_all_apag_compressions(all_apag_data, image_names)

