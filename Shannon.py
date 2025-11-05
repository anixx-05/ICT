#shannon_fano
import os  # For file and folder operations

# Recursive function to generate Shannon-Fano codes
def shannon_fano_recursive(symbols_freq, code=''):
    if len(symbols_freq) == 1:
        return {symbols_freq[0][0]: code}
    
    total_freq = sum(freq for _, freq in symbols_freq)
    cumulative_freq, split_point = 0, 0
    
    for i, (_, freq) in enumerate(symbols_freq):
        cumulative_freq += freq
        if cumulative_freq >= total_freq / 2:
            split_point = i + 1
            break
    
    left = shannon_fano_recursive(symbols_freq[:split_point], code + '0')
    right = shannon_fano_recursive(symbols_freq[split_point:], code + '1')
    
    left.update(right)
    return left

# Encode the message using Shannon-Fano codes
def encode_message(message, codes):
    return ''.join(codes[ch] for ch in message)

# Decode the encoded message using reverse lookup
def decode_message(encoded_message, codes):
    reverse_codes = {v: k for k, v in codes.items()}
    current_code, decoded = '', ''
    
    for bit in encoded_message:
        current_code += bit
        if current_code in reverse_codes:
            decoded += reverse_codes[current_code]
            current_code = ''
    
    return decoded

# Main driver function
def main():
    input_file = input("Enter the text file name: ").strip()

    # Read original message
    with open(input_file, 'r', encoding='utf-8') as file:
        message = file.read().strip()
    print("Original message successfully loaded.")

    # Frequency analysis
    symbols_freq = {char: message.count(char) for char in set(message)}
    symbols_freq = sorted(symbols_freq.items(), key=lambda x: x[1], reverse=True)

    # Generate Shannon-Fano codes
    codes = shannon_fano_recursive(symbols_freq)

    # Encode message
    encoded_message = encode_message(message, codes)

    # Create output folder
    folder_name = "output_files"
    os.makedirs(folder_name, exist_ok=True)

    # Save encoded message
    encoded_file = os.path.join(folder_name, 'encoded_message.fano')
    with open(encoded_file, 'w', encoding='utf-8') as file:
        file.write(encoded_message)
    print(f"Encoded message saved to {encoded_file}")

    # Decode message
    decoded_message = decode_message(encoded_message, codes)

    # Save decoded message
    decoded_file = os.path.join(folder_name, 'decoded_message.txt')
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
    print(f"Compression Ratio: {input_size}:{encoded_size_bytes}:1    ({compression_ratio_value:.2f})")
    print(f"Redundancy: {redundancy:.2f}%")
    print(f"Efficiency: {efficiency:.2f}%")

    # Integrity check
    if message == decoded_message:
        print("\nSUCCESS! Original and decoded messages are EQUAL.")
    else:
        print("\nERROR! Original and decoded messages are NOT equal.")

# Entry point
if __name__ == "__main__":
    main()
