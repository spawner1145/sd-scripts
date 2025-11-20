import os
from PIL import Image, ImageDraw
import numpy as np

def create_dummy_data():
    os.makedirs("dummy_data_simple/img/10_circle", exist_ok=True)
    
    # Create a simple image: Red circle on white background
    img = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([256, 256, 768, 768], fill="red", outline="black")
    
    img.save("dummy_data_simple/img/10_circle/test.png")
    
    with open("dummy_data_simple/img/10_circle/test.txt", "w") as f:
        f.write("a red circle")
        
    print("Simple dummy data created in dummy_data_simple/")

if __name__ == "__main__":
    create_dummy_data()
