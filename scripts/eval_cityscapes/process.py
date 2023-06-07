import os
import Image

def load_images(path):
    lis_dir = os.listdir(path)
    filtered = [file for file in lis_dir if "color" in file]
    
    images = []
    labels = {}
    n = 0

    for file_name in filtered:
        # Create a dictonary that remembers the original file name and temporary filename
        temp_name = f"{n}_B.png"
        labels[temp_name] = file_name
        n += 1

        # Load an image
        img = Image.open(path + file_name)
        images.append(img)
        

    return images, labels
    
def main():
    model = torch.load("latest_model.pth")
    model.eval()

    # do some image loading and tensor
    output = model()