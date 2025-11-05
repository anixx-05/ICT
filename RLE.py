#run length encoding

# Function to perform Run-Length Encoding
def run_length_encode(message):
    encoded = []
    count = 1

    for i in range(1, len(message)):
        if message[i] == message[i - 1]:
            count += 1
        else:
            encoded.append(f"{count}{message[i - 1]}")
            count = 1

    # Append the last run
    encoded.append(f"{count}{message[-1]}")
    return ''.join(encoded)

# Function to decode Run-Length Encoded message
def run_length_decode(encoded_message):
    decoded = []
    count = ''

    for char in encoded_message:
        if char.isdigit():
            count += char
        else:
            decoded.append(char * int(count))
            count = ''

    return ''.join(decoded)

# Main function to drive encoding, decoding, and metrics
def main():
    input_file = input("Enter the text file name: ").strip()

    # Read input message from file
    with open(input_file, 'r', encoding='utf-8') as file:
        message = file.read().strip()

    # Encode the message using RLE
    encoded_message = run_length_encode(message)

    # Save encoded message to file
    encoded_file = 'encoded_message.rle'
    with open(encoded_file, 'w', encoding='utf-8') as file:
        file.write(encoded_message)
    print(f"Encoded message saved to {encoded_file}")

    # Reload encoded message from file
    with open(encoded_file, 'r', encoding='utf-8') as file:
        encoded_message = file.read().strip()

    # Decode the message
    decoded_message = run_length_decode(encoded_message)

    # Save decoded message to file
    decoded_file = 'decoded_message.txt'
    with open(decoded_file, 'w', encoding='utf-8') as file:
        file.write(decoded_message)
    print(f"Decoded message saved to {decoded_file}")

    # Compression metrics
    input_size = len(message.encode("utf-8"))  # Original size in bytes
    encoded_size = len(encoded_message.encode("utf-8"))  # Encoded size in bytes
    decoded_size = len(decoded_message.encode("utf-8"))  # Decoded size in bytes

    print(f"\nOriginal File Size: {input_size} bytes")
    print(f"Encoded File Size: {encoded_size} bytes")
    print(f"Decoded File Size: {decoded_size} bytes")

    if encoded_size > 0:
        compression_ratio = input_size / encoded_size
        redundancy = abs((encoded_size - input_size) / input_size) * 100
        efficiency = (1 - (encoded_size / input_size)) * 100  # Can be negative
        print(f"Compression Ratio: {compression_ratio:.2f} : 1")
        print(f"Redundancy: {redundancy:.2f}%")
        print(f"Efficiency: {efficiency:.2f}%")
    else:
        print("Encoded size is zero. Cannot compute compression metrics.")

    # Integrity check
    if message == decoded_message:
        print("SUCCESS! The decoded message matches the original.")
    else:
        print("ERROR! The decoded message does not match the original.")

# Entry point
if __name__ == "__main__":
    main()
