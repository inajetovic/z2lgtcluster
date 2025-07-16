import pennylane as qml

def chain_hamiltonian(n_fermions, J, m, μ):
    n_qubits = 2*n_fermions - 1

    hamiltonian = 0

    # electric field term
    for i in range(1, n_qubits, 2):
        hamiltonian += qml.PauliX(i) * -1*μ
    
    # mass term
    for i in range(0, n_qubits, 2):
        hamiltonian += (qml.PauliZ(i) + qml.I(i)) * m * 0.5 * (-1)**(i/2)

    # kinetic term
    for i in range(1, n_qubits-1, 2):
        kin_term = (qml.PauliX(i-1) + 1j*qml.PauliY(i-1)) @ qml.PauliZ(i) @ (qml.PauliX(i+1) - 1j*qml.PauliY(i+1))
        #kin_term -= (qml.PauliX(i+1) + 1j*qml.PauliY(i+1)) @ qml.PauliZ(i) @ (qml.PauliX(i-1) - 1j*qml.PauliY(i-1))
        kin_term = kin_term - qml.adjoint(kin_term)
        hamiltonian += J * -1j * kin_term 

    return hamiltonian

### helper functions for obtaining correct ladder indices ###

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


##################################################

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

def id_hamiltonian(ind):
    return qml.I(ind)


def square_hamiltonian(node_dict, J, m, μ, η, V, defects):
    rows, cols = 3, 3

    fermion_nodes = {(0, j) for j in range(cols)} | {(rows-1, j) for j in range(cols)} | \
                 {(i, 0) for i in range(rows)} | {(i, cols-1) for i in range(rows)}

    fermion_dict={k:v for k,v in node_dict.items() if k in fermion_nodes}
    fermion_dict[(1,1)]=4
    link_dict= {k:v for k,v in node_dict.items() if k not in fermion_dict.keys()}

    def neighbours(a,b):
        nb=(a+.5,b),(a-.5,b),(a,b-.5),(a,b+.5)
        return [el for el in nb if el in node_dict]
    
    four_terms=[
        [(.5,0),(0,.5),(1,.5),(.5,1)],
        [(1.5,0),(1,.5),(1.5,1),(2,.5)],
        [(.5,1),(0,1.5),(.5,2),(1,1.5)],
        [(1.5,1),(1,1.5),(1.5,2),(2,1.5)],
    ]

    hamiltonian = []

    # electric field term
    for i in link_dict.values():
        hamiltonian += qml.X(i) * -1*μ

    # mass term
    for i in fermion_dict.values():
        if i%2 == 0:
            hamiltonian += qml.PauliZ(i) + qml.I(i) * m * 0.5 
        else: 
            hamiltonian += qml.PauliZ(i) + qml.I(i) * m * 0.5 * -1

    # magnetic field term
    for el in four_terms:
        term=[node_dict[k] for k in el]
        hamiltonian += 1*η* qml.PauliZ(term[0]) @ qml.PauliZ(term[1]) @ qml.PauliZ(term[2]) @ qml.PauliZ(term[3]) 

    # kinetic energy term
    for a, b in link_dict:
        fermion_pair = [node_dict[i] for i in neighbours(a,b)]
        fermion_pair.sort()
        transverse_fermions = list(range(fermion_pair[0]+1, fermion_pair[1]))
        if not transverse_fermions:
            kin_term =  ((qml.PauliX(fermion_pair[0]) + 1j*qml.PauliY(fermion_pair[0])) 
                        @ qml.PauliZ(node_dict[a,b]) 
                        @ (qml.PauliX(fermion_pair[1]) - 1j*qml.PauliY(fermion_pair[1])))
            kin_term += ((qml.PauliX(fermion_pair[0]) - 1j*qml.PauliY(fermion_pair[0])) 
                        @ qml.PauliZ(node_dict[a,b])
                        @ (qml.PauliX(fermion_pair[1]) + 1j*qml.PauliY(fermion_pair[1])))
        else:
            kin_term =  ((qml.PauliX(fermion_pair[0]) + 1j*qml.PauliY(fermion_pair[0])) 
                    @ qml.PauliZ(node_dict[a,b])
                    @ qml.PauliZ(transverse_fermions[0]) 
                    @ qml.PauliZ(transverse_fermions[1]) 
                    @ (qml.PauliX(fermion_pair[1]) - 1j*qml.PauliY(fermion_pair[1]))) * (-1)
            kin_term += ((qml.PauliX(fermion_pair[0]) - 1j*qml.PauliY(fermion_pair[0])) 
                    @ qml.PauliZ(node_dict[a,b])
                    @ qml.PauliZ(transverse_fermions[0]) 
                    @ qml.PauliZ(transverse_fermions[1]) 
                    @ (qml.PauliX(fermion_pair[1]) + 1j*qml.PauliY(fermion_pair[1]))) * (+1)
        hamiltonian += 0.25*J*kin_term# *1j

    # penalty/charges term 
    matter_sites = [[node_dict[a,b]]+[node_dict[i] for i in neighbours(a,b)] for a,b in fermion_dict]
    hamiltonian += V * gauss_penalty(matter_sites, defects)
    
    return hamiltonian


import networkx as nx
G = nx.Graph()
# Lattice size
rows, cols = 3, 3

# Function to add a node and return its index
def add_node(position, node_dict):
    if position not in node_dict:
        node_dict[position] = len(node_dict)
        G.add_node(node_dict[position], pos=position)
    return node_dict[position]

# Dictionary to track nodes
node_dict = {}

# Adding original 3x3 grid nodes
for i in range(rows):
    for j in range(cols):
        add_node((i, j), node_dict)

# Adding edges with extra nodes
for i in range(rows):
    for j in range(cols):
        if j < cols - 1:  # Horizontal edges
            u = node_dict[(i, j)]
            v = node_dict[(i, j + 1)]
            mid = add_node((i, j + 0.5), node_dict)  # Extra node in between
            G.add_edges_from([(u, mid), (mid, v)])
        
        if i < rows - 1:  # Vertical edges
            u = node_dict[(i, j)]
            v = node_dict[(i + 1, j)]
            mid = add_node((i + 0.5, j), node_dict)  # Extra node in between
            G.add_edges_from([(u, mid), (mid, v)])

#defects are ordered [-1,+1,-1,+1,-1,+1,-1,+1,-1]
# defects = [-1,+1,+1,-1,-1,+1,+1,-1]#,-1]#[0] * 6
# defects = [1, , 0, 0, 0, -1]
# H= ladder_hamiltonian(2,1,1,1,1,1,defects)
# print(qml.simplify(H))
# print(H.simplify())
# print(get3bterms(2))