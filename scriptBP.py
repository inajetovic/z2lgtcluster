import argparse
import pennylane as qml
from pennylane import numpy as np
import networkx as nx
import pickle
import os

##############################
def zzz(theta, sites):
    qml.PauliRot(theta, "ZZZ", wires=sites)

def xzx(theta, sites):
    qml.PauliRot(theta, "XZX", wires=sites)

def yzy(theta, sites):
    qml.PauliRot(theta, "YZY", wires=sites)

def zzzz(theta, sites):
    qml.PauliRot(theta, "ZZZZ", wires=sites)

def xzzy(theta, sites):
    qml.PauliRot(theta, "XZZY", wires=sites)

def yzzx(theta, sites):
    qml.PauliRot(theta, "YZZX", wires=sites)

def layer_ladder(n_plaq, params, j):
    n_qubits = 6*n_plaq + 3
    n_links = 3*n_plaq + 1
    n_fermions = 2*n_plaq + 2
    #s_params = n_links+3*n_fermions+3*n_plaq+1
    s_params = 2*n_links+3*n_fermions+n_plaq

    tbterms = get3bterms(n_plaq)

    for i in range(1, n_qubits, 2):
        c = int(i/2)
        qml.RX(params[c + j*s_params], wires=i)

    allowed_qubits = list(range(0, n_qubits, 2))
    avoided_qubits = [q*6+4 for q in range(0, n_plaq)]
    allowed_qubits = [q for q in allowed_qubits if q not in avoided_qubits]
    
    for i, q in enumerate(allowed_qubits): 
        qml.RX(params[i+n_links + j*s_params], wires=q)
        qml.RY(params[i+n_links+n_fermions + j*s_params], wires=q)
        qml.RZ(params[i+n_links+2*n_fermions + j*s_params], wires=q)

    for i, term in enumerate(tbterms):
        zzz(params[i+n_links+3*n_fermions+ j*s_params], term)

    for i, n in enumerate(range(1, n_qubits-5, 6)):
        zzzz(params[i+2*n_links+3*n_fermions+ j*s_params], [n, n+2, n+4, n+6])


def layer_hva(n_plaq, params, j):
    n_qubits = 6*n_plaq + 3
    n_links = 3*n_plaq + 1
    n_fermions = 2*n_plaq + 2
    s_params = n_links+n_fermions+7*n_plaq+2

    tbterms = get3bterms(n_plaq)

    for i, term in enumerate(tbterms):
        if i < n_plaq+1:
            xzx(params[i+n_links+n_fermions+ j*s_params], term)
            yzy(params[i+n_links+n_fermions+3*n_plaq+1+ j*s_params], term)
        elif i > n_plaq and i < 2*n_plaq+1:
            term.insert(1, term[1]-1)
            xzzy(params[i+n_links+n_fermions+ j*s_params], term)
            yzzx(params[i+n_links+n_fermions+3*n_plaq+1+ j*s_params], term)
        else:
            term.insert(2, term[1]+1)
            xzzy(params[i+n_links+n_fermions+ j*s_params], term)
            yzzx(params[i+n_links+n_fermions+3*n_plaq+1+ j*s_params], term)


    for i, n in enumerate(range(1, n_qubits-5, 6)):
        zzzz(params[i+n_links+n_fermions+6*n_plaq+2+ j*s_params], [n, n+2, n+4, n+6])
    
    for i in range(1, n_qubits, 2):
        c = int(i/2)
        qml.RX(params[c + j*s_params], wires=i)

    allowed_qubits = list(range(0, n_qubits, 2))

    avoided_qubits = [q*6+4 for q in range(0, n_plaq)]
    allowed_qubits = [q for q in allowed_qubits if q not in avoided_qubits]
    
    for i, q in enumerate(allowed_qubits): 
       # qml.RX(params[i+n_links + j*s_params], wires=q)
       # qml.RY(params[i+n_links+n_fermions + j*s_params], wires=q)
        qml.RZ(params[i+n_links + j*s_params], wires=q)

def G_l(node):
    penalty = qml.PauliZ(node[0])
    for i in range(len(node[1:])):
        penalty @= qml.PauliX(node[i+1])
    return penalty


def gauss_penalty(neighbours, defects):
    '''
    Defects are the charges on matter qubits, 
    e.g. for charge at site 2 and 6 matter qubits 
    defects = [1, 1, -1, 1, 1, 1]

    update: if no penalty term, 
    defects = [0, 0, -1, 0, 0, 0]
    also, I think sign of charge changes if mass site is +/-
        so need to add this

    So far, consistent way to place penalties seems to be:
        [-,+,+,-,-,+,...]
    Then charges take opposite signs
    '''
    penalties = []
 
    for i, node in enumerate(neighbours):
        if defects[i] > 0:
            penalties += (2*qml.Identity(node[0]) - 2*G_l(node))#*(1/len(node[1:])) #qml.Identity(node[0])
        if defects[i] < 0:
            penalties += (2*qml.Identity(node[0]) + 2*G_l(node))#*(1/len(node[1:]))
            #2*qml.Identity(node[0]) + 2*G_l(node) #(G_l(node)-qml.Identity(node))@qml.adjoint(G_l(node)-qml.Identity(node))
            #penalties -= G_l(node)@G_l(node)
    
    return penalties

def charges(neighbours, defects):
    '''
    Defects are the charges on matter qubits, 
    e.g. for charge at site 2 and 6 matter qubits 
    defects = [1, 1, -1, 1, 1, 1]

    update: if no penalty term, 
    defects = [0, 0, -1, 0, 0, 0]
    also, I think sign of charge changes if mass site is +/-
        so need to add this

    So far, consistent way to place penalties seems to be:
        [-,+,+,-,-,+,...]
    Then charges take opposite signs
    '''
    penalties = []
 
    for i, node in enumerate(neighbours):
        if defects[i] > 0:
            penalties += G_l(node)#*(1/len(node[1:])) #qml.Identity(node[0])
        if defects[i] < 0:
            penalties += -1*G_l(node)#*(1/len(node[1:]))
            #2*qml.Identity(node[0]) + 2*G_l(node) #(G_l(node)-qml.Identity(node))@qml.adjoint(G_l(node)-qml.Identity(node))
            #penalties -= G_l(node)@G_l(node)
    
    return penalties


def get3bterms(n_plaq):
    # loop over groups '0, 1, 2', '6, 7, 8', '12, 13, 14' ...
    #                  '0, 3, 6', '9, 12, 15', ...
    #                  '2, 5, 8', '11, 14, 17', ...
    terms = []
    n_qubits = 3 + 6*n_plaq

    for i in range(1, n_qubits-1, 6):
        terms.append([i-1,i,i+1])

    for i in range(3, n_qubits-1, 6):
        terms.append([i-3,i,i+3])

    for i in range(5, n_qubits-1, 6):
        terms.append([i-3,i,i+3])

    return terms

def chequer(tbterms, n_plaq):
    signs = []
    terms = tbterms[:n_plaq+1]
    for i, term in enumerate(terms):
        if i % 2 == 0:
            signs.append([1,1,-1])
        #    term[2] = term[2] * -1
        else:
            signs.append([-1,1,1])
        #    term[0] = term[0] * -1
    return signs, terms

def neighbours(n_plaq):
    neighbouring_qubits = []

    _, terms = chequer(get3bterms(n_plaq), n_plaq)

    for i, qubit in enumerate(terms):
        if i == 0:
            neighbouring_qubits.append([qubit[0],qubit[1],qubit[0]+3])
            neighbouring_qubits.append([qubit[2],qubit[1],qubit[2]+3])
        elif i == n_plaq:
            neighbouring_qubits.append([qubit[0],qubit[0]-3,qubit[1]])
            neighbouring_qubits.append([qubit[2],qubit[2]-3,qubit[1]])
        else:
            neighbouring_qubits.append([qubit[0],qubit[0]-3,qubit[1],qubit[0]+3])
            neighbouring_qubits.append([qubit[2],qubit[2]-3,qubit[1],qubit[2]+3])

    return neighbouring_qubits

def ladder_hamiltonian(n_plaq, J, m, μ, η, V, defects):
    n_qubits = 3 + 6*n_plaq
    hamiltonian = 0
    tbterms = get3bterms(n_plaq)
    avoided_qubits = [q*6+4 for q in range(0, n_plaq)]

    # mass terms
    signs, _ = chequer(tbterms, n_plaq)
    for i, signs in enumerate(signs):
        hamiltonian += (qml.PauliZ(tbterms[i][0]) + qml.I(tbterms[i][0])) * m * 0.5 * signs[0]
        hamiltonian += (qml.PauliZ(tbterms[i][2]) + qml.I(tbterms[i][2])) * m * 0.5 * signs[2] 

    # electric field terms
    for i in range(1, n_qubits, 2):
        hamiltonian += qml.PauliX(i) * -1*μ #0.5*μ**2

    # plaquette terms
    for i in range(1, n_qubits-3, 6):
        hamiltonian += 1*η* qml.PauliZ(i) @ qml.PauliZ(i+2) @ qml.PauliZ(i+4) @ qml.PauliZ(i+6) #(1/(2*μ**2))

    # kinetic terms
    for i, term in enumerate(tbterms):
        #long_terms = qml.I(term[0])@qml.I(term[1])@qml.I(term[2])
        for j in range(term[0]+1,term[2]-1):
            if j%2 == 0 and j not in avoided_qubits:
                long_terms = qml.Z(j) # have implemented this way since only have one Z
                long_terms *= 1j
        if term[1] - term[0] == 1:
            kin_term =  -1j *(qml.PauliX(term[0]) + 1j*qml.PauliY(term[0])) @ qml.PauliZ(term[1]) @ (qml.PauliX(term[2]) - 1j*qml.PauliY(term[2]))
            kin_term += 1j *(qml.PauliX(term[0]) - 1j*qml.PauliY(term[0])) @ qml.PauliZ(term[1]) @ (qml.PauliX(term[2]) + 1j*qml.PauliY(term[2]))
            hamiltonian += J*0.25* kin_term
        else:
            kin_term =  -1j *(qml.PauliX(term[0]) + 1j*qml.PauliY(term[0])) @ qml.PauliZ(term[1]) @ long_terms @ (qml.PauliX(term[2]) - 1j*qml.PauliY(term[2]))
            kin_term +=  -1j *(qml.PauliX(term[0]) - 1j*qml.PauliY(term[0])) @ qml.PauliZ(term[1]) @ long_terms @ (qml.PauliX(term[2]) + 1j*qml.PauliY(term[2]))
            hamiltonian += J*0.25* kin_term
        '''#kin_term -= long_terms @ (qml.PauliX(term[2]) + 1j*qml.PauliY(term[2])) @ qml.PauliZ(term[1]) @ (qml.PauliX(term[0]) - 1j*qml.PauliY(term[0]))
        kin_term = kin_term - qml.adjoint(kin_term)
        #kin_term = kin_term * J #* -1j
        hamiltonian += kin_term'''

    matter_sites = neighbours(n_plaq)
    hamiltonian += V * gauss_penalty(matter_sites, defects)

    return hamiltonian

def indexes_zzz(n_qubits):
    list_zzz=[]
    for i in range(1,n_qubits-1,6):
        list_zzz.append([i-1,i,i+1])
    for i in range(3,n_qubits-1,6):
        list_zzz.append([i-3,i,i+3])
    for i in range(5,n_qubits-1,6):
        list_zzz.append([i-3,i,i+3])
    return list_zzz

def create_ladder_lattice(n):
    G = nx.Graph()
    for i in range(n + 1):
        G.add_node((i, 0))
        G.add_node((i, 1))
        G.add_edge((i, 0), (i, 1))
        if i > 0:
            G.add_edge((i - 1, 0), (i, 0))  # bottom
            G.add_edge((i - 1, 1), (i, 1))  # top
    return G

def circuit_state(n_plaq, n_layers, n_qubits, H, charge_inds, hva=False):
    #print(f"Running circuit with {n_layers} layers and {n_qubits} qubits")
    def pqc(params):
        for i in range(1, n_qubits-1, 2):
            qml.H(wires=i)

        fermions = list(range(0, n_qubits, 2))
        avoided_qubits = [q*6+4 for q in range(0, n_plaq)]
        fermions = [q for q in fermions if q not in avoided_qubits]
        odd_fermions1 = [q*12 for q in range(0, n_plaq)]
        odd_fermions2 = [q*12+8 for q in range(0, n_plaq) if q*12+8<=n_qubits]
        #print(odd_fermions2)
        odd_fermions = [item for sublist in zip(odd_fermions1, odd_fermions2) for item in sublist]
    
        if hva:
            for i in odd_fermions:
                qml.X(wires=i)
            if charge_inds != []:
                qml.X(wires=fermions[charge_inds[0]])
                qml.X(wires=fermions[charge_inds[1]])
            for j in range(n_layers):
                layer_hva(n_plaq, params, j)
        else:
            for j in range(n_layers):
                layer_ladder(n_plaq, params, j)
            
        #return qml.state()
        return qml.expval(H)
    return pqc

def return_linkqubits_fermions(n):
    g=create_ladder_lattice(n)
    n_link_qubits=len(g.edges())
    n_ferm_qubits=len(g.nodes())
    return n_link_qubits,n_ferm_qubits

def generic_ansatz2d(nr_qubits,ham_matrix):
    def circuit(parameters):
        qml.StronglyEntanglingLayers(weights=parameters, wires=range(nr_qubits))
        return qml.expval(ham_matrix)
    return circuit

def main(plaquettes, n_layers_list, num_samples):
    np.random.seed(0)
    variances_dict = {"Invariant": {}, "mbqc": {}}

    for n_layers in n_layers_list:
        variances_ours = []
        variances_mbqc = []
        for n_p in plaquettes:
            print(f"Layers: {n_layers}, Plaquettes: {n_p}")
            grad_vals_ours = []
            grad_vals_mbqc = []
            for _ in range(num_samples):
                num_qubits = 3 + 6 * n_p
                n_links, n_fermions = return_linkqubits_fermions(n_p)
                defects = [0] * n_fermions
                hamiltonian = ladder_hamiltonian(n_p, J=3, m=1, μ=2, η=1, V=0, defects=defects)
                dev = qml.device("lightning.qubit", wires=num_qubits)

                # Ours (HVA)
                rand_circuit = circuit_state(n_p, n_layers, num_qubits, hamiltonian, [], True)
                qcircuit = qml.QNode(rand_circuit, dev, interface="autograd")
                grad = qml.grad(qcircuit, argnum=0)
                num_params = n_layers * (n_links + n_fermions + 7 * n_p + 2)
                params = [np.random.randint(-300, 300) / 100 for _ in range(num_params)]
                gradient = grad(params)
                grad_vals_ours.append(gradient[-1])

                # MBQC-style
                rand_circuit_mbqc = circuit_state(n_p, n_layers, num_qubits, hamiltonian, [], False)
                qcircuit_mbqc = qml.QNode(rand_circuit_mbqc, dev, interface="autograd")
                grad_mbqc = qml.grad(qcircuit_mbqc, argnum=0)
                num_params = n_layers * (2 * n_links + 3 * n_fermions + n_p)
                params = [np.random.randint(-300, 300) / 100 for _ in range(num_params)]
                gradient = grad_mbqc(params)
                grad_vals_mbqc.append(gradient[-1])

            variances_ours.append(np.var(grad_vals_ours))
            variances_mbqc.append(np.var(grad_vals_mbqc))
        variances_dict["Invariant"][n_layers] = np.array(variances_ours)
        variances_dict["mbqc"][n_layers] = np.array(variances_mbqc)
    print(variances_dict)

    # Saving the dictionary

    output_file = f"variances_nlayers_{'_'.join(map(str, n_layers_list))}_plaquettes_{'_'.join(map(str, plaquettes))}_samples_{num_samples}.pkl"
    output_path = os.path.join("results", output_file)
    os.makedirs("results", exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(variances_dict, f)

    print(f"\nSaved variances_dict to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ladder circuit gradient variance analysis.")
    parser.add_argument('--plaquettes', type=int, nargs='+', required=True,
                        help='List of plaquette values (e.g. --plaquettes 1 2 3)')
    parser.add_argument('--n_layers_list', type=int, nargs='+', required=True,
                        help='List of layer values (e.g. --n_layers_list 1 2 3)')
    parser.add_argument('--num_samples', type=int, required=True,
                        help='Number of random samples for variance estimation')

    args = parser.parse_args()
    main(args.plaquettes, args.n_layers_list, args.num_samples)