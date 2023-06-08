import os
import tensorflow as tf

# Load image from the original dataset
# Return list of tensors
def load_images(path):
    lis_dir = os.listdir(path)
    filtered = [file for file in lis_dir if "color" in file]
    
    images = []
    name_dict = {}
    n = 0

    for file_name in filtered:
        # Create a dictonary that remembers the original file name and temporary filename
        temp_name = f"{n}_B.png"
        name_dict[n] = (file_name, temp_name)
        n += 1

        # Load an image
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        images.append(img)
        
    return images, name_dict

def preprocess():
    # Do sth similar to preprocess
    
def main():
    model = torch.load("latest_model.pth")
    model.eval()

    # do some image loading and tensor
    IMAGE_PATH = ""
    images, name_dict = load_images(IMAGE_PATH)
    preprocess

    for img
        output = model(img)