def isMatrix(matrix):
    """Returns True if the array given is a valid matrix."""
    if not all(isinstance(i, list) for i in matrix):
        return False
    cols = len(matrix[0])
    for row in matrix:
        if len(row) != cols:
            return False
    return True

def transpose(matrix):
    """Returns the transpose of the matrix given."""
    if not isMatrix(matrix):
        raise ValueError("The array given is not a matrix.")
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def shape(matrix):
    """Returns the shape of the matrix given."""
    if not isMatrix(matrix):
        raise ValueError("The array given is not a matrix.")
    return (len(matrix), len(matrix[0]))

def size(matrix):
    """Returns the size of the matrix given."""
    if not isMatrix(matrix):
        raise ValueError("The array given is not a matrix.")
    return len(matrix) * len(matrix[0])

def identity(n):
    """Returns an identity matrix of size n."""
    return [[1 if i == j else 0 for i in range(n)] for j in range(n)]

def adjoint(matrix):
    """Returns the adjoint of the matrix given."""
    if not isMatrix(matrix):
        raise ValueError("The array given is not a matrix.")
    if len(matrix) != len(matrix[0]):
        raise ValueError("The matrix given is not square.")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return [[matrix[1][1], -matrix[0][1]], [-matrix[1][0], matrix[0][0]]]
    return [[(-1)**(i+j) * minor(matrix, i, j) for j in range(len(matrix))] for i in range(len(matrix))]

def adjugate(matrix):
    """Returns the adjugate of the matrix given."""
    if not isMatrix(matrix):
        raise ValueError("The array given is not a matrix.")
    if len(matrix) != len(matrix[0]):
        raise ValueError("The matrix given is not square.")
    return transpose(adjoint(matrix))

def minor(matrix, row, col):
    """Returns the minor of the matrix given."""
    if not isMatrix(matrix):
        raise ValueError("The array given is not a matrix.")
    if len(matrix) != len(matrix[0]):
        raise ValueError("The matrix given is not square.")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return determinant([row[:col] + row[col+1:] for row in (matrix[:row]+matrix[row+1:])])

def cofactor(matrix, row, col):
    """Returns the cofactor of the matrix given."""
    if not isMatrix(matrix):
        raise ValueError("The array given is not a matrix.")
    if len(matrix) != len(matrix[0]):
        raise ValueError("The matrix given is not square.")
    return (-1) ** (row + col) * minor(matrix, row, col)

def determinant(matrix):
    """Returns the determinant of the matrix given."""
    if not isMatrix(matrix):
        raise ValueError("The array given is not a matrix.")
    if len(matrix) != len(matrix[0]):
        raise ValueError("The matrix given is not square.")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return sum(matrix[0][i] * cofactor(matrix, 0, i) for i in range(len(matrix)))

def inverse(matrix):
    """Returns the inverse of the matrix given."""
    if not isMatrix(matrix):
        raise ValueError("The array given is not a matrix.")
    if len(matrix) != len(matrix[0]):
        raise ValueError("The matrix is not square.")
    if len(matrix) == 1:
        return [[1 / matrix[0][0]]]
    if len(matrix) == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return [[matrix[1][1] / det, -matrix[0][1] / det], [-matrix[1][0] / det, matrix[0][0] / det]]
    else:
        det = 0
        for i in range(len(matrix)):
            det += matrix[0][i] * cofactor(matrix, 0, i)
        return [[cofactor(matrix, i, j) / det for j in range(len(matrix[0]))] for i in range(len(matrix))]

def multiply(a, b):
    """Returns the product of two matrices."""
    if not isMatrix(a) or not isMatrix(b):
        raise ValueError("The array given is not a matrix.")
    if len(a[0]) != len(b):
        raise ValueError("The matrices cannot be multiplied.")
    return [[sum(a[i][k] * b[k][j] for k in range(len(a[0]))) for j in range(len(b[0]))] for i in range(len(a))]

def divide(a, b):
    """Returns the quotient of two matrices."""
    if not isMatrix(a) or not isMatrix(b):
        raise ValueError("The array given is not a matrix.")
    if len(a) != len(a[0]) or len(b) != len(b[0]):
        raise ValueError("The matrices given are not square.")
    if len(a) != len(b):
        raise ValueError("The matrices are not the same size.")
    return multiply(a, inverse(b))

def add(a, b):
    """Returns the sum of two matrices."""
    if not isMatrix(a) or not isMatrix(b):
        raise ValueError("The array given is not a matrix.")
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("The matrices are not the same size.")
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def subtract(a, b):
    """Returns the difference of two matrices."""
    if not isMatrix(a) or not isMatrix(b):
        raise ValueError("The array given is not a matrix.")
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("The matrices are not the same size.")
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]