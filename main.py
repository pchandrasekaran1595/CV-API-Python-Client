import os
import re
import io
import sys
import cv2
import base64
import requests
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

READ_PATH = "Files"


def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")


def decode_image(imageData) -> np.ndarray:
    _, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
    return image


def show(image: np.ndarray) -> None:
    Image.fromarray(image).show()


def draw_box(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    breaker()

    args_1: tuple = ("--url", "-u")
    args_2: tuple = ("--file", "-f")
    args_3: tuple = ("--image-url", "-img-u")
    args_4: tuple = ("--display", "-disp")
    
    url: str = None
    filename: str = None
    image_url: str = None
    display: bool = False

    if args_1[0] in sys.argv: url = sys.argv[sys.argv.index(args_1[0]) + 1]
    if args_1[1] in sys.argv: url = sys.argv[sys.argv.index(args_1[1]) + 1]

    if args_2[0] in sys.argv: filename = sys.argv[sys.argv.index(args_2[0]) + 1]
    if args_2[1] in sys.argv: filename = sys.argv[sys.argv.index(args_2[1]) + 1]

    if args_3[0] in sys.argv: image_url = sys.argv[sys.argv.index(args_3[0]) + 1]
    if args_3[1] in sys.argv: image_url = sys.argv[sys.argv.index(args_3[1]) + 1]

    if args_4[0] in sys.argv or args_4[1] in sys.argv: display = True

    assert url is not None, "No endpoint provided"

    if filename is not None:
        assert filename in os.listdir(READ_PATH), "File not found"
        files = {'image': open(os.path.join(READ_PATH, filename), 'rb')}
        image = cv2.cvtColor(src=cv2.imread(os.path.join(READ_PATH, filename)), code=cv2.COLOR_BGR2RGB)
    
    if image_url is not None:
        response = requests.request("GET", image_url)
        files = {'image': response.content}
        image = np.array(Image.open(io.BytesIO(response.content)))
    
    mode = url.split("/")[-2]
    
    if re.match(r"^classify$", mode, re.IGNORECASE):
        response = requests.request("POST", url=url, files=files)
        if response.status_code == 200:
            label = response.json()["label"]

            print(f"Label: {label}")
            if display: show(image, title=f"Label: {label}")
        else:
            print(f"Error {response.status_code} : {response.reason}")
    
    elif re.match(r"^detect$", mode, re.IGNORECASE):
        response = requests.request("POST", url=url, files=files)
        if response.status_code == 200:
            label = response.json()["label"]

            print(f"Label: {label}")
            if display: 
                draw_box(image, int(response.json()["x1"]), 
                                             int(response.json()["y1"]),
                                             int(response.json()["x2"]),
                                             int(response.json()["y2"]))
                show(image)
        else:
            print(f"Error {response.status_code} : {response.reason}")
        
    elif re.match(r"^segment$", mode, re.IGNORECASE):
        response = requests.request("POST", url=url, files=files)
        if response.status_code == 200:
            labels = response.json()["labels"]

            print(f"Labels: {labels}")
            if display: 
                show(decode_image(response.json()["imageData"]))
        else:
            print(f"Error {response.status_code} : {response.reason}")
    
    else:
        print("Invalid Endpoint")

    breaker()


if __name__ == '__main__':
    sys.exit(main() or 0)
