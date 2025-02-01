from ufl import cos, pi, as_vector


def sinusodial(x):
    return as_vector(
        (
            cos(pi * x[1]),
            cos(pi * x[2]),
            cos(pi * x[0]),
        )
    )

def quadratic(x):
    return as_vector(
        (
            x[1]**2,
            x[2]**2,
            x[0]**2,
        )
    )   