import argparse
import os

import cv2

parser = argparse.ArgumentParser("simple_example")
parser.add_argument('-f', help="Folder with images to resize", type=str, required=True)
parser.add_argument('-o', help="Output folder with images ", type=str, required=True)
args = parser.parse_args()

path = os.path.abspath(args.f if not args.f.endswith('/') else args.f[:-1])

if not os.path.exists(path):
    print(f'Folder {path} not found')
    exit(-1)
print(f'Folder with images: {path}')
listFiles = os.listdir(path)
print(f'Find {len(listFiles)} file to convert')

outputFolder = os.path.abspath(args.o)
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

idx = 0

for f in listFiles:
    filePath = os.path.join(path, f)

    img = cv2.imread(filePath, -1)
    res = cv2.resize(img, (256, 256))
    cv2.imwrite(f'{outputFolder}/{f}', res)
    idx += 1

    print(f'Image resized [{idx}/{len(listFiles)}]')
