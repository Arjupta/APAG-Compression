# Import Required Libraries
import numpy as np
import cv2
import heapq
from collections import defaultdict
import struct

# Generate Quantization Matrix

def get_quantization_matrix(quality_factor=50):
    base_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    scale = 50 / quality_factor
    quant_matrix = np.round(base_matrix * scale).astype(np.int32)
    
    quant_matrix[quant_matrix == 0] = 1
    return quant_matrix

def compute_dct_and_quantize(image, block_size=8, quality_factor=50):
    h, w = image.shape
    final_out = np.zeros_like(image, dtype=np.int32)
    quant_matrix = get_quantization_matrix(quality_factor)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            final_out[i:i+block_size, j:j+block_size] = np.round((cv2.dct(np.float32(block))) / quant_matrix).astype(np.int32)
    return final_out

# Huffman

class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(data):
    ### Returns the Huffman table (dictionary of numbers to Huffman codes) and encoded data as a bitstring.
    # Count frequencies
    freq = defaultdict(int)
    for value in data:
        freq[value] += 1
    
    # Build priority queue
    priority_queue = [HuffmanNode(freq=f, symbol=s) for s, f in freq.items()]
    heapq.heapify(priority_queue)
    
    # Build Huffman Tree
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = HuffmanNode(left.freq + right.freq, left=left, right=right)
        heapq.heappush(priority_queue, merged)
    
    huffman_tree = priority_queue[0]
    huffman_code = {}
    
    # Generate codes
    def generate_code(node, current_code=""):
        if node.symbol is not None:
            huffman_code[node.symbol] = current_code
            return
        generate_code(node.left, current_code + "0")
        generate_code(node.right, current_code + "1")
    
    generate_code(huffman_tree)
    
    # Return the Huffman code table and encoded data (as a bitstring)
    encoded_data = "".join(huffman_code[value] for value in data)
    return huffman_code, encoded_data

def zigzag_order(block):
    h, w = block.shape
    result = np.zeros(h * w, dtype=block.dtype)
    index = -1
    bound = h + w - 1
    for s in range(bound):   # sum of row and col index
        if s % 2 != 0:
            for i in range(s + 1):
                j = s - i
                if i < h and j < w:
                    index += 1
                    # print(i,j)
                    result[index] = block[i, j]
        else:
            for j in range(s + 1):
                i = s - j
                if i < h and j < w:
                    index += 1
                    # print(i,j)
                    result[index] = block[i, j]
    return result

def zigzag_order_all_blocks(quantized_coefficients, block_size=8):
    h, w = quantized_coefficients.shape
    zigzag_result = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = quantized_coefficients[i:i+block_size, j:j+block_size]
            zigzag_result.extend(zigzag_order(block))
    return np.array(zigzag_result)

def run_length_encoding(data):
    rle = []
    counter = 0
    
    for element in data:
        if (element == 0):                 
            counter += 1
        else:                                  
            if (counter != 0):
                rle.extend([0, counter])
                counter = 0
            rle.append(element)
            
    if counter != 0:
        rle.extend([0, counter])
    return rle

# converting the huffman table to bytes to store in file

def save_dict_to_bytearray(data_dict):
    byte_array = bytearray()
    for key, bitstring in data_dict.items():
        # Pack the integer key as a 4-byte signed integer
        byte_array.extend(struct.pack('i', key))  # 'i' is for signed int (4 bytes)
        
        # Convert the bitstring to a byte object
        num_bits = len(bitstring)
        byte_data = int(bitstring, 2).to_bytes((num_bits + 7) // 8, byteorder='big')

        # Write the number of bits (for reading later) followed by the actual byte data
        byte_array.extend(struct.pack('I', num_bits))  # write the number of bits (4 bytes)
        byte_array.extend(byte_data)  # write the actual data as bytes

    return byte_array

def load_dict_from_bytearray(byte_array):
    data_dict = {}
    index = 0
    while index < len(byte_array):
        # Unpack the integer key (4 bytes)
        key = struct.unpack('i', byte_array[index:index+4])[0]
        index += 4
        
        # Unpack the number of bits (4 bytes)
        num_bits = struct.unpack('I', byte_array[index:index+4])[0]
        index += 4
        
        # Read the actual byte data
        num_bytes = (num_bits + 7) // 8
        byte_data = byte_array[index:index+num_bytes]
        index += num_bytes
        
        # Convert the byte data back to a bitstring
        bitstring = bin(int.from_bytes(byte_data, byteorder='big'))[2:].zfill(num_bits)
        
        data_dict[key] = bitstring

    return data_dict

def write_compressed_data_to_file(filename, Q, huffman_table, huffman_encoded_data, h, w):
    with open(filename, 'wb') as file:
        # Write the Q value (4 bytes)
        file.write(struct.pack('I', Q))
        
        # Write the dimensions of the image (height and width, each 4 bytes)
        file.write(struct.pack('I', h))
        file.write(struct.pack('I', w))
        
        # Convert the Huffman table to a bytearray
        huffman_table_bytes = save_dict_to_bytearray(huffman_table)
        
        # Write the size of the Huffman table (4 bytes)
        file.write(struct.pack('I', len(huffman_table_bytes)))
        
        # Write the Huffman table
        file.write(huffman_table_bytes)
        
        # Write the size of the Huffman encoded bitstring (4 bytes)
        file.write(struct.pack('I', len(huffman_encoded_data)))
        
        # Pad the bitstring with zeros to make its length a multiple of 8
        # Write the Huffman encoded data
        padding_length = (8 - len(huffman_encoded_data) % 8) % 8
        huffman_encoded_data = huffman_encoded_data + '0' * padding_length
        byte_array = bytearray()
        for i in range(0, len(huffman_encoded_data), 8):
            byte = huffman_encoded_data[i:i+8]
            byte_array.append(int(byte, 2))
        file.write(byte_array)

# ### Converting the compressed file to original from

import numpy as np

def reverse_zigzag_order(zigzag, h, w):
    block_size = 8
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    reversed_blocks = np.zeros((h, w))
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = zigzag[(i * num_blocks_w + j) * block_size * block_size : (i * num_blocks_w + j + 1) * block_size * block_size]
            reversed_block = np.zeros((block_size, block_size))
            index = -1
            bound = block_size + block_size - 1
            for s in range(bound):  # sum of row and col index
                if s % 2 != 0:
                    for x in range(s + 1):
                        y = s - x
                        if x < block_size and y < block_size:
                            index += 1
                            reversed_block[x, y] = block[index]
                else:
                    for y in range(s + 1):
                        x = s - y
                        if x < block_size and y < block_size:
                            index += 1
                            reversed_block[x, y] = block[index]
            reversed_blocks[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = reversed_block
    return reversed_blocks

def read_compressed_data_from_file(filename):
    with open(filename, 'rb') as file:
        # Read the Q value (4 bytes)
        Q = struct.unpack('I', file.read(4))[0]
        
        # Read the dimensions of the image (height and width, each 4 bytes)
        h = struct.unpack('I', file.read(4))[0]
        w = struct.unpack('I', file.read(4))[0]
        
        # Read the size of the Huffman table (4 bytes)
        huffman_table_size = struct.unpack('I', file.read(4))[0]
        
        # Read the Huffman table
        huffman_table_bytes = file.read(huffman_table_size)
        huffman_table = load_dict_from_bytearray(huffman_table_bytes)
        
        # Read the size of the Huffman encoded bitstring (4 bytes)
        huffman_encoded_data_size = struct.unpack('I', file.read(4))[0]
        
        # Read the Huffman encoded bitstring
        byte_array = file.read()
        huffman_encoded_data = ''.join(f'{byte:08b}' for byte in byte_array)[:huffman_encoded_data_size]
        
    return Q, h, w, huffman_table, huffman_encoded_data

def decompress_data(Q, h, w, huffman_table, huffman_encoded_data):
    # Decode the Huffman encoded data
    reverse_huffman_table = {v: k for k, v in huffman_table.items()}
    decoded_data = []
    current_code = ''
    for bit in huffman_encoded_data:
        current_code += bit
        if current_code in reverse_huffman_table:
            decoded_data.append(reverse_huffman_table[current_code])
            current_code = ''
    
    # Perform run-length decoding
    rle_decoded_data = []
    i = 0
    while i < len(decoded_data):
        if decoded_data[i] == 0:
            rle_decoded_data.extend([0] * decoded_data[i + 1])
            i += 2
        else:
            rle_decoded_data.append(decoded_data[i])
            i += 1
    
    # Make h and w multiple of 8
    h_new = (h + 7) // 8 * 8
    w_new = (w + 7) // 8 * 8
    reversed_zigzag = reverse_zigzag_order(rle_decoded_data, h_new, w_new)
    
    # Dequantize the DCT coefficients
    quant_matrix = get_quantization_matrix(Q)
    
    # Perform inverse DCT
    h2, w2 = reversed_zigzag.shape
    block_size = quant_matrix.shape[0]
    image = np.zeros_like(reversed_zigzag, dtype=np.float32)
    for i in range(0, h2, block_size):
        for j in range(0, w2, block_size):
            block = reversed_zigzag[i:i + block_size, j:j + block_size]
            block = block * quant_matrix
            image[i:i + block_size, j:j + block_size] = cv2.idct(np.float32(block))
    
    # Crop the image to the original dimensions
    image = image[:h, :w]
    final_img = np.round(image)
    final_img[final_img > 255] = 255
    final_img[final_img < 0] = 0
    return final_img.astype(np.uint8)

def generate_compressed_file(image, output_filename, quality_factor):
    # Pad the image to make its dimensions a multiple of 8
    h, w = image.shape
    new_h = (h + 7) // 8 * 8
    new_w = (w + 7) // 8 * 8
    padded_image = np.full((new_h, new_w), 255, dtype=np.uint8)
    padded_image[:h, :w] = image
    image = padded_image

    quantized_coefficients = compute_dct_and_quantize(image, quality_factor=quality_factor)
    zigzag_coefficients = zigzag_order_all_blocks(quantized_coefficients)
    rle_encoded_data = run_length_encoding(zigzag_coefficients)
    huffman_table, huffman_encoded_data = huffman_encoding(rle_encoded_data)
    write_compressed_data_to_file(output_filename, quality_factor, huffman_table, huffman_encoded_data, h, w)

def decompress_image_from_file(compressed_filename, compressed_image_name = None):
    Q, h, w, huffman_table, huffman_encoded_data = read_compressed_data_from_file(compressed_filename)
    decompressed_image = decompress_data(Q, h, w, huffman_table, huffman_encoded_data)
    if compressed_image_name is not None:
        decompressed_image_path = "../output/apag/decompressed_" + compressed_image_name + ".jpg"
        cv2.imwrite(decompressed_image_path, decompressed_image)
    return decompressed_image

### Color Image Compression

def compress_image(image_path, output_filename, quality_factor):
    color_image = cv2.imread(image_path)
    h, w, _ = color_image.shape
    new_h = (h + 1) // 2 * 2
    new_w = (w + 1) // 2 * 2
    padded_color_image = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
    padded_color_image[:h, :w, :] = color_image
    color_image = padded_color_image

    ycbcr_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2YCrCb)
    Y, Cb, Cr = cv2.split(ycbcr_image)
    Cb_downsampled = cv2.resize(Cb, (Cb.shape[1] // 2, Cb.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    Cr_downsampled = cv2.resize(Cr, (Cr.shape[1] // 2, Cr.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    Cb_Cr_concatenated = np.concatenate((Cb_downsampled, Cr_downsampled), axis=1)
    concatenated_array = np.concatenate((Y, Cb_Cr_concatenated), axis=0)
    generate_compressed_file(concatenated_array, output_filename, quality_factor)

def decompress_image(compressed_filename):
    decompressed_image = decompress_image_from_file(compressed_filename)
    shape_Y = (int(decompressed_image.shape[0]/3 * 2), decompressed_image.shape[1])
    shape_Cr = shape_Cb = (int(decompressed_image.shape[0]/3), int(decompressed_image.shape[1]/2))
    Y = decompressed_image[:shape_Y[0], :]
    Cb_Cr = decompressed_image[shape_Y[0]:, :]
    Cb = Cb_Cr[:, :shape_Cb[1]]
    Cr = Cb_Cr[:, shape_Cb[1]:]
    Cb = cv2.resize(Cb, (shape_Cb[1] * 2, shape_Cb[0] * 2), interpolation=cv2.INTER_LINEAR)
    Cr = cv2.resize(Cr, (shape_Cr[1] * 2, shape_Cr[0] * 2), interpolation=cv2.INTER_LINEAR)
    ycbcr_decompressed = cv2.merge((Y, Cb, Cr))
    color_decompressed = cv2.cvtColor(ycbcr_decompressed, cv2.COLOR_YCrCb2BGR)
    return color_decompressed




# ##### Grayscale Image Compression
# input_image_path = 'input/barbara256.png'
# output_filename = 'output/compressed_output.bin'

# quality_factor = 50
# image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
# generate_compressed_file(image, output_filename, quality_factor)
# decompressed_image_path = 'output/decompressed_image.png'
# decompressed_img = decompress_image_from_file(output_filename)
# cv2.imwrite(decompressed_image_path, decompressed_img)
# #####



# ##### Color Image Compression
# input_image_path = 'input/1.jpg'
# output_filename = 'output/compressed_output_ycbcr.bin'
# quality_factor = 50
# compress_image(input_image_path, output_filename, quality_factor)
# color_decompressed = decompress_image(output_filename)
# cv2.imwrite('output/decompressed_color_image.png', color_decompressed)
# #####