

# from unittest import result


def add_numbers(a, b):
    result = a + b
    return result


def main(x0, y0):
    x1 = x0 * 3
    y1 = y0 * 4
    result = add_numbers(x1, y1)
    return result


if __name__ == "__main__":
    x0 = 1
    y0 = 2
    result = main(x0, y0)
    print(result)