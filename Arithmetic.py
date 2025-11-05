#arithmetic

from collections import defaultdict
from decimal import Decimal, getcontext
import os

# Set high precision for decimal operations
getcontext().prec = 500

# Calculate symbol ranges based on frequency
def calculate_ranges(message):
    frequency = defaultdict(int)
    for char in message:
        frequency[char] += 1
    total_chars = len(message)
    ranges = {}
    lower_bound = Decimal(0)
    for char, count in frequency.items():
        ranges[char] = (
            lower_bound / total_chars,
            (lower_bound + count) / total_chars
        )
        lower_bound += count
    return ranges

# Perform Arithmetic Encoding
def arithmetic_encode(message):
    ranges = calculate_ranges(message)
    low = Decimal(0.0)
    high = Decimal(1.0)

    for char in message:
        range_width = high - low
        high = low + range_width * Decimal(ranges[char][1])
        low = low + range_width * Decimal(ranges[char][0])
    return (low + high) / 2  # Final encoded value

# Decode using Arithmetic Decoding
def arithmetic_decode(encoded_value, message, ranges):
    low = Decimal(0.0)
    high = Decimal(1.0)
    decoded_message = ""
    for _ in range(len(message)):
        range_width = high - low
        value = (encoded_value - low) / range_width
        for char, (low_range, high_range) in ranges.items():
            if Decimal(low_range) <= value < Decimal(high_range):
                decoded_message += char
                high = low + range_width * Decimal(high_range)
                low = low + range_width * Decimal(low_range)
                break
    return decoded_message

# Main driver function
def main():
    input_file = input("Enter the text file name: ").strip()

    # Read input message
    with open(input_file, 'r', encoding='utf-8') as file:
        message = file.read().strip()

    # Encode message
    encoded_value = arithmetic_encode(message)

    # Save encoded value
    encoded_file = 'encoded_value.arith'
    with open(encoded_file, 'w', encoding='utf-8') as file:
        file.write(str(encoded_value))
    print(f"Encoded value saved to {encoded_file}")

    # Decode message
    ranges = calculate_ranges(message)
    decoded_message = arithmetic_decode(Decimal(encoded_value), message, ranges)

    # Save decoded message
    decoded_file = 'decoded_message.txt'
    with open(decoded_file, 'w', encoding='utf-8') as file:
        file.write(decoded_message)
    print(f"Decoded message saved to {decoded_file}")

    # Compression metrics
    input_size = len(message.encode("utf-8"))  # Original size in bytes
    encoded_size_bytes = len(str(encoded_value).encode("utf-8"))  # Encoded value size in bytes
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
    print(f"Encoded Value Size: {encoded_size_bytes} bytes")
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
if __name__ == "__main__":  # Corrected syntax
    main()