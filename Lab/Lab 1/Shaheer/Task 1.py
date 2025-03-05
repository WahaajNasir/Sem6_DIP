def normalize_min_max(values):
    highest = max(values)
    lowest = min(values)

    return [(num - lowest) / (highest - lowest) for num in values]


size = int(input("Enter the number of elements: "))
data = [int(input(f"Enter value {i}: ")) for i in range(size)]

normalized_data = normalize_min_max(data)
print("Normalized Data:", normalized_data)
