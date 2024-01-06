import matplotlib.patches as patches
import matplotlib.pyplot as plt
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
        # plt.text(bbox[0], bbox[1], label, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()


def extract_bbox(text):
    # Find all matches
    matches = re.findall(r'\[\s*(?:\d+(?:\.\d+)?\s*(?:,\s*)?)+\]', text)

    # Extract the full matched strings (first group in each match)
    extracted_lists = [json.loads(match) for match in matches]

    return extracted_lists


if __name__ == "__main__":
    visualize_bounding_boxes('./data/DIOR-RSVG/JPEGImages/05093.jpg', [[ 549, 289, 584, 304 ]], ['vehicle'])
