#dic

# Initialize base dictionary with ASCII characters
def initialize_dictionary():
    return {chr(i): i for i in range(256)}

# Function to perform Dictionary-based Encoding (LZW)
def lzw_encode(message):
    dictionary = initialize_dictionary()
    current_string = ""
    encoded_output = []

    for char in message:
        combined_string = current_string + char
        if combined_string in dictionary:
            current_string = combined_string
        else:
            encoded_output.append(dictionary[current_string])
            dictionary[combined_string] = len(dictionary)
            current_string = char

    if current_string:
        encoded_output.append(dictionary[current_string])

    return encoded_output

# Function to decode LZW encoded output
def lzw_decode(encoded_output):
    dictionary = initialize_dictionary()
    reverse_dict = {v: k for k, v in dictionary.items()}
    decoded_message = ""
    prev_entry = ""

    for code in encoded_output:
        if code in reverse_dict:
            entry = reverse_dict[code]
        elif code == len(reverse_dict):
            entry = prev_entry + prev_entry[0]
        else:
            raise ValueError("Invalid LZW code encountered.")

        decoded_message += entry

        if prev_entry:
            reverse_dict[len(reverse_dict)] = prev_entry + entry[0]

        prev_entry = entry

    return decoded_message

# Main function to drive encoding, decoding, and metrics
def main():
    input_file = input("Enter the text file name: ").strip()

    # Read input message from file
    with open(input_file, 'r', encoding='utf-8') as file:
        message = file.read().strip()

    # Encode the message using LZW
    encoded_output = lzw_encode(message)

    # Save encoded output to file
    encoded_file = 'encoded_message.lzw'
    with open(encoded_file, 'w', encoding='utf-8') as file:
        file.write(','.join(map(str, encoded_output)))
    print(f"Encoded message saved to {encoded_file}")

    # Reload encoded output from file
    with open(encoded_file, 'r', encoding='utf-8') as file:
        encoded_output = list(map(int, file.read().split(',')))

    # Decode the message
    decoded_message = lzw_decode(encoded_output)

    # Save decoded message to file
    decoded_file = 'decoded_message.txt'
    with open(decoded_file, 'w', encoding='utf-8') as file:
        file.write(decoded_message)
    print(f"Decoded message saved to {decoded_file}")

    # Compression metrics
    input_size = len(message.encode("utf-8"))  # Original size in bytes
    encoded_size_bytes = len(encoded_output) * 2  # Approximate: 2 bytes per code
    decoded_size = len(decoded_message.encode("utf-8"))  # Decoded size in bytes

    print(f"\nOriginal File Size: {input_size} bytes")
    print(f"Encoded Output Size (approx): {encoded_size_bytes} bytes")
    print(f"Decoded File Size: {decoded_size} bytes")

    if encoded_size_bytes > 0:
        compression_ratio = input_size / encoded_size_bytes
        redundancy = abs((encoded_size_bytes - input_size) / input_size) * 100
        print(f"Compression Ratio: {compression_ratio:.2f} : 1")
        print(f"Redundancy: {redundancy:.2f}%")
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