# save as extract_intra_coeffs.py
import cv2
import numpy as np
import os
import time

class IntraDCTExtractor:
    def _init_(self, qp=5):
        self.block_size = 8
        self.macroblock_size = 16
        self.qp = qp
        self.quant_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ]).astype(np.float32)

    def dct2d(self, block):
        return cv2.dct(block.astype(np.float32))

    def idct2d(self, block):
        return cv2.idct(block.astype(np.float32))

    def quantize(self, dct_block):
        q_table = self.quant_table * (self.qp / 16.0)
        q_table[q_table == 0] = 1.0
        return np.round(dct_block / q_table).astype(np.int16)

    def dequantize(self, quant_block):
        q_table = self.quant_table * (self.qp / 16.0)
        return quant_block.astype(np.float32) * q_table

    def process_first_intra_frame(self, input_file, output_folder='output', target_width=352):
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open input file: {input_file}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # compute target size aligned to macroblocks
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        aligned_width = (target_width // self.macroblock_size) * self.macroblock_size
        aligned_height = (target_height // self.macroblock_size) * self.macroblock_size

        # read first frame (frame 0 -> intra)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Cannot read first frame.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (aligned_width, aligned_height))

        # subtract 128 for intra residual (as in your original code)
        residual_frame = gray.astype(np.float32) - 128.0

        dct_blocks = []
        quant_blocks = []

        for y in range(0, aligned_height, self.macroblock_size):
            for x in range(0, aligned_width, self.macroblock_size):
                # process each macroblock as 4 blocks of 8x8
                for by in range(0, self.macroblock_size, self.block_size):
                    for bx in range(0, self.macroblock_size, self.block_size):
                        block = residual_frame[y + by:y + by + self.block_size,
                                               x + bx:x + bx + self.block_size]
                        if block.shape != (self.block_size, self.block_size):
                            # pad if necessary (shouldn't happen due aligned sizes)
                            block = cv2.copyMakeBorder(block, 0, self.block_size - block.shape[0],
                                                      0, self.block_size - block.shape[1],
                                                      cv2.BORDER_CONSTANT, value=0)
                        dct_block = self.dct2d(block)
                        quant_block = self.quantize(dct_block)

                        dct_blocks.append(dct_block.astype(np.float32))
                        quant_blocks.append(quant_block.astype(np.int16))

        cap.release()

        dct_array = np.stack(dct_blocks, axis=0)      # shape: (N_blocks, 8, 8)
        quant_array = np.stack(quant_blocks, axis=0)  # shape: (N_blocks, 8, 8)

        out_path = os.path.join(output_folder, f'intr_coeffs_frame0_qp{self.qp}.npz')
        np.savez_compressed(out_path, dct_blocks=dct_array, quant_blocks=quant_array,
                            aligned_width=aligned_width, aligned_height=aligned_height,
                            qp=self.qp, num_blocks=dct_array.shape[0])
        print(f"Saved intra-frame coefficients to: {out_path}")
        print(f"Number of 8x8 blocks saved: {dct_array.shape[0]}")
        return out_path, dct_array, quant_array

# ---------------------------
# Simple runner
# ---------------------------
if _name_ == "_main_":
    INPUT_FILE = 'sample1.avi'   # change if needed
    OUTPUT_FOLDER = 'output'
    QP_VALUE = 5

    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found. Place your video in the same folder or update the path.")
    else:
        extractor = IntraDCTExtractor(qp=QP_VALUE)
        start = time.time()
        out_path, dct_array, quant_array = extractor.process_first_intra_frame(INPUT_FILE, OUTPUT_FOLDER)
        elapsed = time.time() - start
        print(f"Done in {elapsed:.2f} s.")
        # quick example: print first block's matrices
        print("\nFirst 8x8 DCT coefficients (float):\n", dct_array[0])
        print("\nFirst 8x8 Quantized coefficients (int):\n", quant_array[0])