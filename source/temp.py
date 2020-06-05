def sum2(a, b):
    return lambda c : sum3(a, b, c)
    pass

def sum3(a, b, c):
    return a + b + c


a = sum2(1, 2)
b = a(2)
print(b)