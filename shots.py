import pennylane as qml
from pennylane import numpy as np
from hamiltonian import ladder_hamiltonian
from ansatz import circuit
import pandas as pd
import argparse


def run_vqe(n_plaq, n_layers, J, m, mu, V, charge_inds, n_iter, shots,invariant=True):
    n_fermions = 2*n_plaq + 2
    n_links = 3*n_plaq + 1
    n_qubits = 6*n_plaq + 3

    if invariant:
        s_params = n_links+n_fermions+7*n_plaq+2 #hva
    else:
        s_params = 2*n_links+3*n_fermions+n_plaq

    n_params = n_layers*s_params

    if n_plaq==2:
        ordering = [-1,+1,+1,-1,-1,+1]
    else:
        ordering = [-1,+1,+1,-1,-1,+1,+1,-1]

    #charges = ordering
    for i, c in enumerate(charge_inds):
        sign = ordering[c] 
        ordering[c] = sign * -1

    H = ladder_hamiltonian(n_plaq, J, m, mu, 1, V, ordering)    

    dev = qml.device("default.qubit", wires=n_qubits, shots=shots) 

    @qml.qnode(dev)
    def cost(params):
        circuit(n_plaq, n_layers, n_qubits, H, params, charge_inds, invariant)
        return qml.expval(H)

    
    opt = qml.SPSAOptimizer(maxiter=n_iter,a=.2)
    params =  np.array([np.random.random(1)[0] for _ in range(n_params)], requires_grad=True)

    for _ in range(n_iter):
        params, energy = opt.step_and_cost(cost, params)
    return energy

def main(n_plaquettes, n_layers, shots, iterations):
    np.random.seed(0)

    df=pd.DataFrame()
    for n_plaq in n_plaquettes:
        for n_l in n_layers:
            for n_shots in shots:
                for n_iter in iterations:
                    for flag in [False,True]:
                        if flag:
                            circuit="GI"
                        else:
                            circuit="ZZ"
                        
                        energies=run_vqe(n_plaq, n_l, 3,1,2,0,[], n_iter=n_iter,shots=n_shots,invariant=flag)
                        
                        new_row={
                            "P":n_plaq,
                            "L":n_l,
                            "S":n_shots,
                            "Ansatz":circuit,
                            "E":energies.real
                           
                        }
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True) 
    # Save variances_dict to CSV
    csv_output_path = f"SPSA_{n_plaquettes}_layers_{n_layers}_shots_{n_shots}_iters_{iterations}.csv"
    df.to_csv(csv_output_path, index=False) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPSA Bench")
    parser.add_argument('--n_plaquettes', type=int, nargs='+', required=True,
                        help='List of plaquette values (e.g. --n_plaquettes 1 2 3)')
    parser.add_argument('--n_layers', type=int, nargs='+', required=True,
                        help='List of layer values (e.g. --n_layers_list 1 2 3)')
    parser.add_argument('--shots', type=int,nargs='+', required=True,
                        help='Number of shots')
    parser.add_argument('--iterations', type=int,nargs='+',  required=True,
                        help='iterations for optimizer')

    args = parser.parse_args()
    main(args.n_plaquettes, args.n_layers, args.shots, args.iterations)
