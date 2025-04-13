import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
img = np.array([
    [10, 20, 30, 40, 50, 60, 70, 80],
    [15, 25, 35, 45, 55, 65, 75, 85],
    [20, 30, 40, 50, 60, 70, 80, 90],
    [25, 35, 45, 55, 65, 75, 85, 95],
    [30, 40, 50, 60, 70, 80, 90,100],
    [35, 45, 55, 65, 75, 85, 95,105],
    [40, 50, 60, 70, 80, 90,100,110],
    [45, 55, 65, 75, 85, 95,105,115]
], dtype=float)

#missing pixels
i1, j1 = 3, 3
i2, j2 = 3, 4
img[i1, j1] = 0
img[i2, j2] = 0

plt.imshow(img, cmap='gray', vmin=10, vmax=115)
plt.title("Image with Missing Pixels at (3,3) and (3,4)")
plt.colorbar()
plt.show()

A = np.array([
    [4, -1],  # for (3,3), 4x1 - xtop - xbottom - xleft - x2 = 0 -> 4x1 - x2 = sum of known neighbors
    [-1, 4]   # for (3,4), 4x2 - xtop - xbottom - x1 - xright = 0 -> 4x2 - x1 = sum of known neighbors
])

b = np.array([
    [img[2,3] + img[4,3] + img[3,2] + 0],  # 0 is placeholder for missing pixel 
    [img[2,4] + img[4,4] + 0 + img[3,5]]   # 0 is placeholder for missing pixel 
])

# CLASSICAL SOLUTION
x = np.linalg.solve(A, b)

img[i1, j1] = x[0]
img[i2, j2] = x[1]
print(x[0])
print(x[1])

# Show the repaired image
plt.imshow(img, cmap='gray', vmin=10, vmax=115)
plt.title("Image After Inpainting Two Pixels")
plt.colorbar()
plt.show()

##Quantum Solution###
n_qubits = 3  # Number of system qubits.
n_shots = 10 ** 6  # Number of quantum measurements.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 30  # Number of optimization steps
eta = 0.8  # Learning rate
q_delta = 0.001  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator

c = np.array([1.0, 0.2, 0.2])

def U_b():
    """Unitary matrix rotating the ground"""
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

def CA(idx):
    """Controlled versions of the unitary components A_l of the problem matrix A."""
    if idx == 0:
        None

    elif idx == 1:
        qml.CNOT(wires=[ancilla_idx, 0])
        qml.CZ(wires=[ancilla_idx, 1])

    elif idx == 2:
        qml.CNOT(wires=[ancilla_idx, 0])


def variational_block(weights):
    """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

    for idx, element in enumerate(weights):
        qml.RY(element, wires=idx)


dev_mu = qml.device("lightning.qubit", wires=tot_qubits)

@qml.qnode(dev_mu, interface="autograd")
def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):

    qml.Hadamard(wires=ancilla_idx)

    if part == "Im" or part == "im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

    variational_block(weights)
    CA(l)
    U_b()

    if j != -1:
        qml.CZ(wires=[ancilla_idx, j])

    U_b()

    CA(lp)
    qml.Hadamard(wires=ancilla_idx)
    return qml.expval(qml.PauliZ(wires=ancilla_idx))

def mu(weights, l=None, lp=None, j=None):

    mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")

    return mu_real + 1.0j * mu_imag

def psi_norm(weights):
    norm = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)

    return abs(norm)

def cost_loc(weights):
    mu_sum = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)

    mu_sum = abs(mu_sum)

    return 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))

np.random.seed(rng_seed)
w = q_delta * np.random.randn(n_qubits, requires_grad=True)

opt = qml.GradientDescentOptimizer(eta)
cost_history = []
for it in range(steps):
    w, cost = opt.step_and_cost(cost_loc, w)
    print("Step {:3d}       Cost_L = {:9.7f}".format(it, cost))
    cost_history.append(cost)

Id = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

A_0 = np.identity(8)
A_1 = np.kron(np.kron(X, Z), Id)
A_2 = np.kron(np.kron(X, Id), Id)

A_num = c[0] * A_0 + c[1] * A_1 + c[2] * A_2
b = np.ones(8) / np.sqrt(8)
print("A = \n", A_num)
print("b = \n", b)

dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)

@qml.qnode(dev_x, interface="autograd")
def prepare_and_sample(weights):
    
    variational_block(weights)
    return qml.sample()


def extract_solution_vector(probabilities, n_qubits):
    dim = 2 ** n_qubits
    probs = np.zeros(dim)
    probs[:len(probabilities)] = probabilities  
    amplitudes = np.sqrt(probs)

    norm = np.linalg.norm(amplitudes)
    if norm == 0:
        return amplitudes
    return amplitudes / norm



raw_samples = prepare_and_sample(w)

samples = []
for sam in raw_samples:
    samples.append(int("".join(str(bs) for bs in sam), base=2))

q_probs = np.bincount(samples, minlength=2**n_qubits) / n_shots
print(q_probs)
x = extract_solution_vector(q_probs, n_qubits)

print("Quantum Solution vector x:")
print(x)

A_inv = np.linalg.inv(A_num)
x = np.dot(A_inv, b)
print("Classical Solution vector x:")
print(x)