import math

import numpy

# region Input Training Set process
print('\n--------------------------------------------------------------------\nDataSet\n{')
data_set = numpy.loadtxt(r'C:\Users\User\OneDrive\Desktop\Model.txt', usecols=0)
print(data_set)
print('}\n--------------------------------------------------------------------')
numberOfObjects = len(data_set)
numberOfFeatures = 1
real_values = numpy.array(numpy.loadtxt(r'C:\Users\User\OneDrive\Desktop\Model.txt', usecols=numberOfFeatures))
print("Javoblar To'plami:\n", real_values, "\n--------------------------------------------------------------------")


# endregions

# region Functions
def mean_squared_error_comparer(current_vector, next_vector):
    mean_squared_error_current = sum(current_vector ** 2) / (2 * len(current_vector))
    mean_squared_error_next = sum(next_vector ** 2) / (2 * len(next_vector))
    return math.fabs(mean_squared_error_next - mean_squared_error_current) < epsilion


# endregion

# region Setting initial values
data_set = numpy.column_stack([[1] * numberOfObjects, data_set])
current_coefficients_of_weight = [0] * (numberOfFeatures + 1)
next_coefficients_of_weights = []
current_estimated_values = numpy.dot(current_coefficients_of_weight, data_set.transpose())
current_vector_of_errors = numpy.subtract(real_values, current_estimated_values)
epsilion = float(input("Epsilionga qiymat beirng:"))
linear_rate = float(input("O'garish tezligini kiriting:"))
continue_value = True
# endregion

# region Phase of building a good model
iterations = 0
while continue_value:
    iterations += 1
    gradient_vector = numpy.dot(current_vector_of_errors, data_set) / numberOfObjects
    next_coefficients_of_weights = current_coefficients_of_weight + gradient_vector * linear_rate
    next_estimated_values = numpy.dot(next_coefficients_of_weights, data_set.transpose())
    next_vector_of_errors = numpy.subtract(real_values, next_estimated_values)
    if mean_squared_error_comparer(current_vector_of_errors, next_vector_of_errors):
        continue_value = False
    else:
        current_coefficients_of_weight = next_coefficients_of_weights
        current_vector_of_errors = next_vector_of_errors
print(f"Multiple Linear Regression Modelini shakllantirish uchun ketgan iteratsiyalar soni: {iterations}ta")
# endregion.

# region Step of entering test objects
while True:
    test_object = numpy.array(
        [float(temp) for temp in input("Obyekt elementlarini probel bilan ajratib kiriting:").split()])
    print(test_object)
    print(next_coefficients_of_weights)
    result = numpy.dot(next_coefficients_of_weights, test_object)
    print(result)
# endregion
