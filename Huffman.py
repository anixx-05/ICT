#huffman

import heapq  # For priority queue operations

# Node class for Huffman tree
class Node:
    def __init__(self, char, freq):  # Constructor to initialize character and frequency
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):  # Comparison method for heap operations
        return self.freq < other.freq

# Build Huffman tree from symbol frequencies
def build_huffman_tree(symbols_freq):
    heap = [Node(char, freq) for char, freq in symbols_freq]  # Create nodes for each symbol
    heapq.heapify(heap)  # Convert list into a min-heap

    while len(heap) > 1:
        left = heapq.heappop(heap)  # Pop two nodes with lowest frequency
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)  # Create a new internal node
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)  # Push merged node back into heap

    return heap[0]  # Return root of the Huffman tree

# Recursively generate Huffman codes from the tree
def generate_huffman_codes(node, code='', codes={}):
    if node is not None:
        if node.char is not None:  # Leaf node
            codes[node.char] = code
        generate_huffman_codes(node.left, code + '0', codes)
        generate_huffman_codes(node.right, code + '1', codes)
    return codes

# Encode message using Huffman codes
def encode_message(message, codes):
    return ''.join(codes[char] for char in message)

# Decode message using Huffman tree
def decode_message(encoded_message, root):
    decoded_message = ''
    current_node = root

    for bit in encoded_message:
        current_node = current_node.left if bit == '0' else current_node.right
        if current_node.char is not None:
            decoded_message += current_node.char
            current_node = root

    return decoded_message

# Main driver function
def main():
    input_file = input("Enter the text file name: ")
    with open(input_file, 'r', encoding='utf-8') as file:
        message = file.read().strip()
    print("Original message:", message)

    # Frequency analysis
    symbols_freq = {char: message.count(char) for char in set(message)}
    symbols_freq = sorted(symbols_freq.items(), key=lambda x: x[1])  # Sort by frequency (ascending)

    # Build Huffman tree and generate codes
    huffman_tree_root = build_huffman_tree(symbols_freq)
    codes = generate_huffman_codes(huffman_tree_root)

    # Encode message
    encoded_message = encode_message(message, codes)
    print("Encoded Message:", encoded_message)

    # Save encoded message
    encoded_file = 'encoded_message.huff'
    with open(encoded_file, 'w', encoding='utf-8') as file:
        file.write(encoded_message)
    print(f"Encoded message saved to {encoded_file}")

    # Decode message
    decoded_message = decode_message(encoded_message, huffman_tree_root)
    print("Decoded Message:", decoded_message)

    # Save decoded message
    decoded_file = 'decoded_message.txt'
    with open(decoded_file, 'w', encoding='utf-8') as file:
        file.write(decoded_message)
    print(f"Decoded message saved to {decoded_file}")

    # Compression metrics
    input_size = len(message.encode("utf-8"))  # Original size in bytes
    encoded_size_bytes = (len(encoded_message) + 7) // 8  # Approx encoded size in bytes
    decoded_size = len(decoded_message.encode("utf-8"))  # Decoded size in bytes

    if encoded_size_bytes > 0:
        compression_ratio_value = input_size / encoded_size_bytes
        redundancy = (1 - 1 / compression_ratio_value) * 100
        efficiency = 100 - redundancy
    else:
        compression_ratio_value = float('inf')
        redundancy = 100
        efficiency = 0

    # Display metrics
    print(f"\nOriginal File Size: {input_size} bytes")
    print(f"Encoded File Size (approx): {encoded_size_bytes} bytes")
    print(f"Decoded File Size: {decoded_size} bytes")
    print(f"Compression Ratio: {compression_ratio_value:.2f} : 1")
    print(f"Redundancy: {redundancy:.2f}%")
    print(f"Efficiency: {efficiency:.2f}%")

    # Integrity check
    if message == decoded_message:
        print("\nSUCCESS! The decoded message matches the original.")
    else:
        print("\nERROR! The decoded message does not match the original.")

# Entry point
if __name__ == "__main__":  # Ensures the script runs only when executed directly
    main()