import math

# 3D Case
def dot3d(v, w):
    x, y, z = v
    X, Y, Z = w
    return x * X + y * Y + z * Z


def length3d(v):
    x, y, z = v
    return math.sqrt(x * x + y * y + z * z)


def vector3d(b, e):
    x, y, z = b
    X, Y, Z = e
    return X - x, Y - y, Z - z


def unit3d(v):
    x, y, z = v
    mag = length3d(v)
    return x / mag, y / mag, z / mag


def distance3d(p0, p1):
    return length3d(vector3d(p0, p1))


def scale3d(v, sc):
    x, y, z = v
    return x * sc, y * sc, z * sc


def add3d(v, w):
    x, y, z = v
    X, Y, Z = w
    return x + X, y + Y, z + Z


def pnt2line3d(pnt, start, end):
    line_vec = vector3d(start, end)
    pnt_vec = vector3d(start, pnt)
    line_len = length3d(line_vec)
    line_unitvec = unit3d(line_vec)
    pnt_vec_scaled = scale3d(pnt_vec, 1.0 / line_len)
    t = dot3d(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale3d(line_vec, t)
    dist = distance3d(nearest, pnt_vec)
    nearest = add3d(nearest, start)
    return dist, nearest


# 2D Case
def dot2d(v, w):
    x, y = v
    X, Y = w
    return x * X + y * Y


def length2d(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def vector2d(b, e):
    x, y = b
    X, Y = e
    return X - x, Y - y


def unit2d(v):
    x, y = v
    mag = length2d(v)
    return x / mag, y / mag


def distance2d(p0, p1):
    return length2d(vector2d(p0, p1))


def scale2d(v, sc):
    x, y = v
    return x * sc, y * sc


def add2d(v, w):
    x, y = v
    X, Y = w
    return x + X, y + Y


def pnt2line2d(pnt, start, end):
    line_vec = vector2d(start, end)
    pnt_vec = vector2d(start, pnt)
    line_len = length2d(line_vec)
    line_unitvec = unit2d(line_vec)
    pnt_vec_scaled = scale2d(pnt_vec, 1.0 / line_len)
    t = dot2d(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale2d(line_vec, t)
    dist = distance2d(nearest, pnt_vec)
    nearest = add2d(nearest, start)
    return dist, nearest
