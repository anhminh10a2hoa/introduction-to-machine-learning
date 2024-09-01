my_list = [13, 666, -666]

sorted_list1 = sorted(my_list)

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

sorted_list2 = bubble_sort(my_list.copy())

print("Sorted List 1:", sorted_list1)
print("Sorted List 2:", sorted_list2)