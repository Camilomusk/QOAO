!pip install cirq
import cirq
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Definir el problema de Max-Cut
def cost_function_max_cut(bitstring, weights):
    cost = 0
    for i in range(len(bitstring)):
        for j in range(i+1, len(bitstring)):
            if bitstring[i] != bitstring[j]:  # Si los bits son diferentes (se cortan)
                cost += weights[i][j]
    return -cost  # Queremos maximizar el costo, por lo que lo hacemos negativo

# Simulación del circuito QAOA
def create_qaoa_circuit(params, num_qubits):
    gamma, beta = params
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    circuit = cirq.Circuit()

    # Puertas Hadamard para crear superposición
    circuit.append(cirq.H.on_each(qubits))

    # Aquí puedes agregar más puertas dependiendo del problema
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            circuit.append(cirq.CNOT(qubits[i], qubits[j]))

    for i in range(num_qubits):
        circuit.append(cirq.rz(-gamma).on(qubits[i]))

    # Aplicar las rotaciones RX para cada qubit
    for i in range(num_qubits):
        circuit.append(cirq.rx(2 * beta).on(qubits[i]))

    # Medir los qubits
    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit

# Evaluar la energía promedio
def evaluate_cost(params, weights):
    circuit = create_qaoa_circuit(params, len(weights))
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1024)
    measurements = result.measurements['result']
    counts = np.unique([''.join(map(str, row)) for row in measurements], return_counts=True)

    energy = 0
    for bitstring, count in zip(counts[0], counts[1]):
        energy += count * cost_function_max_cut(bitstring, weights)
    return energy / 1024

# Optimizar parámetros
def optimize_qaoa(weights):
    initial_params = [np.pi / 2, np.pi / 4]
    result = minimize(
        evaluate_cost,
        initial_params,
        args=(weights,),
        method='COBYLA',
        options={'maxiter': 100}
    )
    return result.x, result.fun

# Mostrar resultados con visualización
def show_results():
    # Definir pesos para un ejemplo de Max-Cut
    weights = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])  # Grafo simple
    optimal_params, optimal_energy = optimize_qaoa(weights)
    gamma, beta = optimal_params

    print("Parámetros óptimos encontrados:")
    print(f"Gamma: {gamma:.4f}")
    print(f"Beta: {beta:.4f}")
    print("\nEnergía mínima encontrada:")
    print(f"{optimal_energy:.4f}")

    # Graficar los resultados
    plt.bar(["Gamma", "Beta"], [gamma, beta], color='blue')
    plt.title("Parámetros óptimos de QAOA")
    plt.ylabel("Valor")
    plt.show()

    # Graficar la energía durante la optimización (esto será representado en función de los parámetros)
    gammas = np.linspace(0, np.pi, 100)
    betas = np.linspace(0, np.pi, 100)
    energies = np.zeros((len(gammas), len(betas)))

    for i, g in enumerate(gammas):
        for j, b in enumerate(betas):
            energies[i, j] = evaluate_cost([g, b], weights)

    plt.contourf(betas, gammas, energies, levels=50, cmap="inferno")
    plt.colorbar(label="Energía")
    plt.xlabel("Beta")
    plt.ylabel("Gamma")
    plt.title("Mapa de energía de QAOA")
    plt.show()

# Ejecutar
show_results()
