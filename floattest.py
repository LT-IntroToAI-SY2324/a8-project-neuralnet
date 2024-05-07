def closest_value(n):
    # Find the fractional part of the input
    fraction = n % 1
    
    # Define the possible values
    values = [0, 0.3, 0.6, 1]
    
    # Find the closest value
    closest = min(values, key=lambda x: abs(x - fraction))
    
    return closest

# Test the function
print(closest_value(0.1))  # Output: 0.3
print(closest_value(0.4))  # Output: 0.3
print(closest_value(0.7))  # Output: 0.6
print(closest_value(0.8))  # Output: 1
