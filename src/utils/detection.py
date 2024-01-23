import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import re


def visualize_bounding_boxes(image_path, bounding_boxes, annotations):
    """
    Visualizes bounding boxes on an image with text annotations.

    Parameters:
    - image_path: str, the path to the image file.
    - bounding_boxes: list of lists, each containing 4 integers [xmin, ymin, xmax, ymax].
    - annotations: list of str, text annotations for the bounding boxes.
    """
    # Open the image file
    im = Image.open(image_path)
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Check if the number of bounding boxes matches the number of annotations
    if len(bounding_boxes) != len(annotations):
        raise ValueError("The number of bounding boxes must match the number of annotations.")

    # Add the bounding boxes and annotations to the image
    for bbox, label in zip(bounding_boxes, annotations):
        # Create a Rectangle patch
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Add the annotation
        plt.text(bbox[0], bbox[1], label, bbox=dict(facecolor='white', alpha=0.5))
        plt.xticks([])
        plt.yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # plt.show()


def extract_bbox(text):
    # Find all matches
    matches = re.findall(r'\[\s*(?:\d+(?:\.\d+)?\s*(?:,\s*)?)+\]', text)

    # Extract the full matched strings (first group in each match)
    extracted_lists = [json.loads(match) for match in matches]

    return extracted_lists


def extract_qwen_bbox(text, h, w):
    box_pattern = r'<box>\s*(?:\(\d+,\d+\)\s*,?\s*)+</box>'

    # Find all matches for <box> tags
    box_matches_raw = re.findall(box_pattern, text)
    box_matches = []
    for match in box_matches_raw:
        # Removing parentheses and splitting by comma
        numbers = [int(x) for x in re.findall(r'\d+', match)]

        x1, y1, x2, y2 = numbers
        x1, y1, x2, y2 = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
        box_matches.extend([x1, y1, x2, y2])

    return box_matches if len(box_matches) == 4 else []


def get_mean_centroid_distance(bboxes_a, bboxes_b):
    # (xmin, ymin, xmax, ymax)
    x_centroids_a = (bboxes_a[:, 0] + bboxes_a[:, 2]) / 2
    y_centroids_a = (bboxes_a[:, 1] + bboxes_a[:, 3]) / 2

    x_centroids_b = (bboxes_b[:, 0] + bboxes_b[:, 2]) / 2
    y_centroids_b = (bboxes_b[:, 1] + bboxes_b[:, 3]) / 2

    mean_dist = np.sqrt((x_centroids_a - x_centroids_b) ** 2 + (y_centroids_a - y_centroids_b) ** 2).mean()

    return mean_dist


if __name__ == "__main__":
    visualize_bounding_boxes("./data/DIOR-RSVG/JPEGImages/13790.jpg", [[325, 325, 475, 475], [233, 383, 376, 542]], annotations=["GPT-4V", "Human"])
    plt.savefig("./data/DIOR-RSVG/13790_annotate.png", dpi=500)
