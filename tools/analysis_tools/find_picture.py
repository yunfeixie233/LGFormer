import cv2
import numpy as np
import os
import shutil
from heapq import nlargest
from collections import defaultdict

def calculate_histogram(image_path):
    """ Calculate the color histogram of an image. """
    image = cv2.imread(image_path)
    histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram

def find_most_similar_images(folder_path, target_image_paths, top_k=5):
    # Pre-calculate histograms for all images in the folder
    folder_histograms = {}
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        folder_histograms[filename] = calculate_histogram(image_path)

    results = defaultdict(list)

    for target_image_path in target_image_paths:
        target_histogram = calculate_histogram(target_image_path)
        similarities = {}

        for filename, image_histogram in folder_histograms.items():
            similarity = cv2.compareHist(target_histogram, image_histogram, cv2.HISTCMP_CORREL)
            similarities[filename] = similarity

        # Get top-k similar images
        top_k_images = nlargest(top_k, similarities, key=similarities.get)
        results[target_image_path] = [(image, similarities[image]) for image in top_k_images]

    return results

def save_top_k_images(results, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for target_image, similar_images in results.items():
        target_image_name = os.path.basename(target_image)
        target_image_folder = os.path.join(output_folder, os.path.splitext(target_image_name)[0])

        if not os.path.exists(target_image_folder):
            os.makedirs(target_image_folder)

        for image, _ in similar_images:
            src = os.path.join(folder_path, image)
            dst = os.path.join(target_image_folder, image)
            shutil.copy(src, dst)

# Example usage
folder_path = '/data1/yunfei/SpformerV1/data/VOCdevkit/VOC2010/JPEGImages/val'  # 替换为你的文件夹路径
target_image_paths = ['/data1/yunfei/p1.png',
                      '/data1/yunfei/p2.png',
                      '/data1/yunfei/p3.png',
                      '/data1/yunfei/p4.png',
                      '/data1/yunfei/p5.png']   # Replace with your target image paths
output_folder = '/data1/yunfei/output'  # Replace with the path to the folder where you want to save the results

top_k_similar_images = find_most_similar_images(folder_path, target_image_paths, top_k=5)
save_top_k_images(top_k_similar_images, output_folder)

for target_image, images in top_k_similar_images.items():
    print(f"\nTop {len(images)} similar images for {target_image} saved to {output_folder}:")
    for image, similarity in images:
        print(f"{image}, Similarity: {similarity:.4f}")


