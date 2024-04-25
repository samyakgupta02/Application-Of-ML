import numpy as np

# Task 5: Basic NumPy Operations
arr = np.arange(1, 11)
arr2 = np.arange(11, 21)

add_result = arr + arr2
subtract_result = arr - arr2
multiply_result = arr * arr2
divide_result = arr / arr2

print("Task 5 :")
print("Addition:", add_result)
print("Subtraction:", subtract_result)
print("Multiplication:", multiply_result)
print("Division:", divide_result)

# Task 6: Array Manipulation
reshaped_arr = arr.reshape(2, 5)
transposed_arr = reshaped_arr.T
flattened_arr = transposed_arr.flatten()
stacked_arr = np.vstack((arr, arr2))

print("\nTask 6 Results:")
print("Reshaped Array:\n", reshaped_arr)
print("Transposed Array:\n", transposed_arr)
print("Flattened Array:", flattened_arr)
print("Vertically Stacked Array:\n", stacked_arr)

# Task 7: Statistical Operations
mean_arr = np.mean(arr)
median_arr = np.median(arr)
std_dev_arr = np.std(arr)
max_arr = np.max(arr)
min_arr = np.min(arr)
normalized_arr = (arr - mean_arr) / std_dev_arr

print("\nTask 7 Results:")
print("Mean:", mean_arr)
print("Median:", median_arr)
print("Standard Deviation:", std_dev_arr)
print("Max Value:", max_arr)
print("Min Value:", min_arr)
print("Normalized Array:", normalized_arr)

# Task 8: Boolean Indexing
bool_arr = arr > 5
filtered_arr = arr[bool_arr]

print("\nTask 8 Results:")
print("Boolean Array:", bool_arr)
print("Filtered Array:", filtered_arr)

# Task 9: Random Module
random_matrix = np.random.rand(3, 3)
random_integers = np.random.randint(1, 101, 10)
np.random.shuffle(arr)

print("\nTask 9 Results:")
print("Random Matrix:\n", random_matrix)
print("Random Integers:", random_integers)
print("Shuffled Array:", arr)

# Task 10: Universal Functions (ufunc)
sqrt_arr = np.sqrt(arr)
exp_arr = np.exp(arr)

print("\nTask 10 Results:")
print("Square Root of arr:", sqrt_arr)
print("Exponential of arr:", exp_arr)

# Task 11: Linear Algebra Operations
mat_a = np.random.rand(3, 3)
vec_b = np.random.rand(3, 1)
dot_product_result = np.dot(mat_a, vec_b)

print("\nTask 11 Results:")
print("Matrix A:\n", mat_a)
print("Vector B:\n", vec_b)
print("Dot Product Result:\n", dot_product_result)

# Task 12: Broadcasting
matrix = np.arange(1, 10).reshape(3, 3)
row_means = matrix.mean(axis=1, keepdims=True)
broadcasted_result = matrix - row_means

print("\nTask 12 Results:")
print("Original Matrix:\n", matrix)
print("Broadcasted Result:\n", broadcasted_result)
