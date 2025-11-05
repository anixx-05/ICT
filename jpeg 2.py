import numpy as np
import os
from PIL import Image
from scipy.fftpack import dct
import time

# JPEG Standard Quantization Table (Luminance)
QY = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

def process_image_dct_quant(image_path, output_folder='output', block_size=8):
    os.makedirs(output_folder, exist_ok=True)
    start_time = time.time()

    # Output text file to store all results
    log_file_path = os.path.join(output_folder, 'dct_quantization_output.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')

    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape

    # Padding
    padded_h = int(np.ceil(h / block_size) * block_size)
    padded_w = int(np.ceil(w / block_size) * block_size)
    padded_img = np.zeros((padded_h, padded_w), dtype=np.float32)
    padded_img[:h, :w] = img_array

    # Level shift
    padded_img -= 128.0

    block_count = 0
    for y in range(0, padded_h, block_size):
        for x in range(0, padded_w, block_size):
            block = padded_img[y:y+block_size, x:x+block_size]

            # DCT transform
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
