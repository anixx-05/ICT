#lpc

# Convert text to list of ASCII values
def text_to_ascii(text):
    return [ord(char) for char in text]

# Convert list of ASCII values back to text
def ascii_to_text(ascii_list):
    return ''.join(chr(value) for value in ascii_list)

# Perform LPC encoding: store difference from previous value
def lpc_encode(ascii_values):
    prediction = 0
    encoded_list = []
    for value in ascii_values:
        encoded_value = value - prediction
        encoded_list.append(encoded_value)
        prediction = value
    return encoded_list

# Perform LPC decoding: reconstruct original values from differences
def lpc_decode(encoded_values):
    prediction = 0
    decoded_list = []
    for value in encoded_values:
        decoded_value = value + prediction
        decoded_list.append(decoded_value)
        prediction = decoded_value
    return decoded_list

# Full pipeline: read, encode, decode, save, and compute metrics
def text_lpc_pipeline():
    # Step 1: Get input file name from user
    input_text_file = input("Enter the input text file name: ").strip()
    encoded_text_file = 'encoded_message.lpc'
    decoded_text_file = 'decoded_message.txt'

    # Step 2: Read input text from file
    with open(input_text_file, 'r', encoding='utf-8') as file:
        text = file.read().strip()

    # Step 3: Convert text to ASCII values
    ascii_values = text_to_ascii(text)

    # Step 4: Encode using LPC
    encoded_values = lpc_encode(ascii_values)

    # Step 5: Save encoded values to file
    with open(encoded_text_file, 'w', encoding='utf-8') as file:
        file.write(' '.join(map(str, encoded_values)))
    print(f"Encoded values saved to {encoded_text_file}")

    # Step 6: Decode back to ASCII values
    decoded_ascii = lpc_decode(encoded_values)

    # Step 7: Convert decoded ASCII values back to text
    decoded_text = ascii_to_text(decoded_ascii)

    # Step 8: Save decoded text to file
    with open(decoded_text_file, 'w', encoding='utf-8') as file:
        file.write(decoded_text)
    print(f"Decoded text saved to {decoded_text_file}")

    # Step 9: Compression metrics
    input_size = len(text.encode("utf-8"))  # Original size in bytes
    encoded_size = len(' '.join(map(str, encoded_values)).encode("utf-8"))  # Encoded size in bytes
    decoded_size = len(decoded_text.encode("utf-8"))  # Decoded size in bytes

    print(f"\nOriginal File Size: {input_size} bytes")
    print(f"Encoded File Size: {encoded_size} bytes")
    print(f"Decoded File Size: {decoded_size} bytes")

    if encoded_size > 0:
        compression_ratio = input_size / encoded_size
        redundancy = abs((encoded_size - input_size) / input_size) * 100
        print(f"Compression Ratio: {compression_ratio:.2f} : 1")
        print(f"Redundancy: {redundancy:.2f}%")
    else:
        print("Encoded size is zero. Cannot compute compression metrics.")

    # Step 10: Integrity check
    if text == decoded_text:
        print("SUCCESS! The decoded message matches the original.")
    else:
        print("ERROR! The decoded message does not match the original.")

# Entry point
if __name__ == "__main__":
    text_lpc_pipeline()
