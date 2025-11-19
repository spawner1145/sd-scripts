import os
from PIL import Image
import numpy as np

os.makedirs("dummy_data/img/10_dog", exist_ok=True)
img = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
img.save("dummy_data/img/10_dog/test.png")
with open("dummy_data/img/10_dog/test.txt", "w") as f:
    f.write("a photo of a dog")
print("Dummy data created.")
