#simple convolutional neural network (CNN)

from ad import dual, max

def convnet(x):
    """
    Function calculating output of simple CNN using ReLU activation and dual numbers
    """
    

    w1 = dual(1.2, 1)
    w2 = dual(-0.2, 0)
    v1 = dual(-0.3, 0)
    v2 = dual(0.6, 0)
    v3 = dual(1.3, 0)
    v4 = dual(-1.5, 0)
    z = []
    group = [x[i: i + 2] for i in range(len(x) - 2 + 1)]

    for pair in group:
        z.append(max(dual(0, 0), w1 * dual(pair[0], 0) + w2 * dual(pair[1], 0)))

    y = max(dual(0, 0), -(v1 * z[0] + v2 * z[1] + v3 * z[2] + v4 * z[3]))

    return y, z


x = [0.3, -1.5, 0.7, 2.1, 0.1]

print(convnet(x))