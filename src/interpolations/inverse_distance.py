
def inverse_distance_weighting_with_missing(data, power=2, p=2):
    for i in range(len(data)):
        if data[i] == 0.0:
            data_missing = data[i]
            interpolated_value = 0.0
            total_weight = 0

            for j in range(len(data)):
                if data[j] != 0.0:
                    distance = abs(j - i)
                    if distance == 0:
                        # Avoid division by zero
                        interpolated_value = data[j]
                        break
                    weight = 1 / (distance ** power)
                    interpolated_value += data[j] * weight
                    total_weight += weight

            if total_weight != 0:
                interpolated_value /= total_weight

            data[i] = interpolated_value

    return data

#usage 
list1 = [0.0, 0.4, 0.0, 0.0, 0.0, 0.7]
list1 = inverse_distance_weighting_with_missing(list1)
print(list1)