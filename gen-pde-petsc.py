from scipy.sparse import csr_array, diags
from scipy import io

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import numpy as np
import argparse

# Example invocation:
# mpirun -np 1 --bind-to core python3 examples/pde-petsc.py -ksp_type cg -nx 4000 -ny 4000 -ksp_max_it 100
parser = argparse.ArgumentParser()
parser.add_argument("-nx", type=int, default=101)
parser.add_argument("-ny", type=int, default=101)
args, _ = parser.parse_known_args()

nx = args.nx              # number of points in the x direction
ny = args.ny              # number of points in the y direction
xmin, xmax = 0.0, 1.0     # limits in the x direction
ymin, ymax = -0.5, 0.5    # limits in the y direction
lx = xmax - xmin          # domain length in the x direction
ly = ymax - ymin          # domain length in the y direction
dx = lx / (nx-1)          # grid spacing in the x direction
dy = ly / (ny-1)          # grid spacing in the y direction

# Create the gridline locations and the mesh grid;
# see notebook 02_02_Runge_Kutta for more details
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
# We pass the argument `indexing='ij'` to np.meshgrid
# as x and y should be associated respectively with the
# rows and columns of X, Y.
X, Y = np.meshgrid(x, y, indexing='ij')

# Compute the rhs. Note that we non-dimensionalize the coordinates
# x and y with the size of the domain in their respective dire-
# ctions.
b = (np.sin(np.pi*X)*np.cos(np.pi*Y)
     + np.sin(5.0*np.pi*X)*np.cos(5.0*np.pi*Y))

# b is currently a 2D array. We need to convert it to a column-major
# ordered 1D array. This is done with the flatten numpy function.
# We use the parameter 'F' to specify that we want want column-major
# ordering. The letter 'F' is used because this is the natural
# ordering of the popular Fortran language. For row-major
# ordering you can pass 'C' as paremeter, which is the natural
# ordering for the C language.
# More info
# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
bflat = b[1:-1, 1:-1].flatten('F')

# Allocate array for the (full) solution, including boundary values
p = np.empty((nx, ny))

def d2_mat_dirichlet_2d(nx, ny, dx, dy):
    """
    Constructs the matrix for the centered second-order accurate
    second-order derivative for Dirichlet boundary conditions in 2D

    Parameters
    ----------
    nx : integer
        number of grid points in the x direction
    ny : integer
        number of grid points in the y direction
    dx : float
        grid spacing in the x direction
    dy : float
        grid spacing in the y direction

    Returns
    -------
    d2mat : numpy.ndarray
        matrix to compute the centered second-order accurate first-order deri-
        vative with Dirichlet boundary conditions
    """
    a = 1.0 / dx**2
    g = 1.0 / dy**2
    c = -2.0*a - 2.0*g

    # TODO (rohany): We can't do this strided access with cunumeric directly right now. Is there
    #  another way to get the same behavior?
    diag_a = a * np.ones((nx-2)*(ny-2)-1)
    diag_a[nx-3::nx-2] = 0.0
    diag_a = np.array(diag_a)
    diag_g = g * np.ones((nx-2)*(ny-3))
    diag_c = c * np.ones((nx-2)*(ny-2))

    # We construct a sequence of main diagonal elements,
    diagonals = [diag_g, diag_a, diag_c, diag_a, diag_g]
    # and a sequence of positions of the diagonal entries relative to the main
    # diagonal.
    offsets = [-(nx-2), -1, 0, 1, nx-2]

    # Call to the diags routine; note that diags return a representation of the
    # array; to explicitly obtain its ndarray realisation, the call to .toarray()
    # is needed. Note how the matrix has dimensions (nx-2)*(nx-2).
    d2mat = diags(diagonals, offsets, dtype=np.float64).tocsr()

    # Return the final array
    return d2mat


def p_exact_2d(X, Y):
    """Computes the exact solution of the Poisson equation in the domain
    [0, 1]x[-0.5, 0.5] with rhs:
    b = (np.sin(np.pi * X) * np.cos(np.pi * Y) +
    np.sin(5.0 * np.pi * X) * np.cos(5.0 * np.pi * Y))

    Parameters
    ----------
    X : numpy.ndarray
        array of x coordinates for all grid points
    Y : numpy.ndarray
        array of y coordinates for all grid points

    Returns
    -------
    sol : numpy.ndarray
        exact solution of the Poisson equation
    """

    sol = (-1.0/(2.0*np.pi**2)*np.sin(np.pi*X)*np.cos(np.pi*Y)
           - 1.0/(50.0*np.pi**2)*np.sin(5.0*np.pi*X)*np.cos(5.0*np.pi*Y))

    return sol


A = d2_mat_dirichlet_2d(nx, ny, dx, dy)
n = A.shape[0]
A_p = PETSc.Mat().createAIJ(size=(n, n), csr=(A.indptr, A.indices, A.data))
viewer = PETSc.Viewer().createBinary("A.dat", "w")
A_p.view(viewer)
bflat_p = PETSc.Vec().createSeq(n)
bflat_p.setValues(range(n), bflat)
viewer = PETSc.Viewer().createBinary("bflat.dat", "w")
bflat_p.view(viewer)
x = PETSc.Vec().createSeq(n)
x.set(0)
viewer = PETSc.Viewer().createBinary("x.dat", "w")
x.view(viewer)
