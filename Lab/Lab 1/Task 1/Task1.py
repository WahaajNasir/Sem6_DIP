def min_max_normal(my_list):
    max_list = max(my_list)
    min_list = min(my_list)
    scaled_list = []
    for x in my_list:
        x = (x - min_list)/(max_list-min_list)

        scaled_list.append(x)

    return scaled_list

length = int(input("Enter the length of data: "))
data = []
for x in range(length):
    temp = int(input("\nEnter value "+ str(x) + ":"))
    data.append(temp)

normalized = min_max_normal(data)
print(normalized)