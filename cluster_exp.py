
def Fast_Hadamard(a):
    ordering = 1.0       
    a = a.copy()
    N = len(a)
    h = 1
    while h < N:
        # reshape into blocks of size 2h
        a2 = a.reshape(-1, 2*h)

        # slices
        x = a2[:, :h]
        y = a2[:, h:]

        # butterfly
        t = x.copy()
        a2[:, :h] = 0.5 * (t + y)
        a2[:, h:] = 0.5 * (t-  y) * ordering

        h *= 2

    return a
