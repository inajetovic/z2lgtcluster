import pennylane as qml
import numpy as np

from hamiltonian import get3bterms
np.random.seed()

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



def layer_square(node_dict, params, j):
    s_params = 12*1 + 9*3 + 4 + 12
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

    for i in link_dict.values():
        qml.RX(params[i+ j*s_params],wires=i)

    for i in fermion_dict.values():
        qml.RX(params[i+(12)+ j*s_params],wires=i)
        qml.RY(params[i+(12+9)+ j*s_params],wires=i)
        qml.RZ(params[i+(12+2*9)+ j*s_params],wires=i)
    
    for el in four_terms:
        term=[node_dict[k] for k in el]
        zzzz(params[i+(12+3*9)+ j*s_params], term)

    for a,b in link_dict:
        term = [node_dict[a,b]]+[node_dict[i] for i in neighbours(a,b)]
        zzz(params[i+(12+3*9+4)+ j*s_params], term)


def circuit(n_plaq, n_layers, n_qubits, H,params, charge_inds, hva=False):
    #print(f"Running circuit with {n_layers} layers and {n_qubits} qubits")

        for i in range(1, n_qubits-1, 2):
            qml.H(wires=i)

        '''if n_plaq == 1:
            odd_sites = [0, 8]#[2, 6]#
        elif n_plaq == 2:
            odd_sites = [0, 8, 12]#[2, 6, 14]#
        elif n_plaq == 3:
            odd_sites = [0, 8, 12, 20]#[2, 6, 14, 18]#
        else:
            odd_sites = [0, 8, 12, 20, 24]#[2, 6, 14, 18, 26]#'''

        # for i in odd_sites:
        #     qml.X(wires=i)
        """
        fermions = list(range(0, n_qubits, 2))
        avoided_qubits = [q*6+4 for q in range(0, n_plaq)]
        fermions = [q for q in fermions if q not in avoided_qubits]
        odd_fermions1 = [q*12 for q in range(0, n_plaq)]
        odd_fermions2 = [q*12+8 for q in range(0, n_plaq) if q*12+8<=n_qubits]
        odd_fermions = [item for sublist in zip(odd_fermions1, odd_fermions2) for item in sublist]"""

        fermions = list(range(0, n_qubits, 2))
        avoided_qubits = [q*6+4 for q in range(0, n_plaq)]
        fermions = [q for q in fermions if q not in avoided_qubits]
        odd_fermions1 = [q*12 for q in range(0, n_plaq)]
        odd_fermions2 = [q*12+8 for q in range(0, n_plaq)]
        odd_fermions = [item for sublist in zip(odd_fermions1, odd_fermions2) for item in sublist]
        odd_fermions[:] = [x for x in odd_fermions if x < n_qubits]

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

def circuit_p(n_plaq, n_layers, n_qubits, H,params, charge_inds, hva=False):
    #print(f"Running circuit with {n_layers} layers and {n_qubits} qubits")

        for i in range(1, n_qubits-1, 2):
            qml.H(wires=i)

        '''if n_plaq == 1:
            odd_sites = [0, 8]#[2, 6]#
        elif n_plaq == 2:
            odd_sites = [0, 8, 12]#[2, 6, 14]#
        elif n_plaq == 3:
            odd_sites = [0, 8, 12, 20]#[2, 6, 14, 18]#
        else:
            odd_sites = [0, 8, 12, 20, 24]#[2, 6, 14, 18, 26]#'''

        # for i in odd_sites:
        #     qml.X(wires=i)
        fermions = list(range(0, n_qubits, 2))
        avoided_qubits = [q*6+4 for q in range(0, n_plaq)]
        fermions = [q for q in fermions if q not in avoided_qubits]
        odd_fermions1 = [q*12 for q in range(0, n_plaq)]
        odd_fermions2 = [q*12+8 for q in range(0, n_plaq)]
        odd_fermions = [item for sublist in zip(odd_fermions1, odd_fermions2) for item in sublist]
        odd_fermions[:] = [x for x in odd_fermions if x < n_qubits]

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

        return qml.expval(H)

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

def circuit_square(node_dict, n_layers, H, params):
    rows, cols = 3, 3
    n_qubits = 12 + 9
    n_params = len(params)

    fermion_nodes = {(0, j) for j in range(cols)} | {(rows-1, j) for j in range(cols)} | \
                 {(i, 0) for i in range(rows)} | {(i, cols-1) for i in range(rows)}

    fermion_dict={k:v for k,v in node_dict.items() if k in fermion_nodes}
    fermion_dict[(1,1)]=4
    link_dict= {k:v for k,v in node_dict.items() if k not in fermion_dict.keys()}

    for i in link_dict.values():
        qml.H(wires=i)

    for i in fermion_dict.values():
        if i % 2 != 0:
            qml.X(wires=i) #?

    for j in range(n_layers):
        layer_square(node_dict, params, j)

    for i in range(n_qubits):
        qml.RX(params[n_params-1-i], wires=i)

    return qml.expval(H)

def circuit_square_state(node_dict, n_layers):
    rows, cols = 3, 3
    n_qubits = 12 + 9
    def pqc(params):
        n_params = len(params)

        fermion_nodes = {(0, j) for j in range(cols)} | {(rows-1, j) for j in range(cols)} | \
                    {(i, 0) for i in range(rows)} | {(i, cols-1) for i in range(rows)}

        fermion_dict={k:v for k,v in node_dict.items() if k in fermion_nodes}
        fermion_dict[(1,1)]=4
        link_dict= {k:v for k,v in node_dict.items() if k not in fermion_dict.keys()}

        for i in link_dict.values():
            qml.H(wires=i)

        for i in fermion_dict.values():
            if i % 2 != 0:
                qml.X(wires=i) #?

        for j in range(n_layers):
            layer_square(node_dict, params, j)

        for i in range(n_qubits):
            qml.RX(params[n_params-1-i], wires=i)
        return qml.state()
    return pqc

### code bin 

'''params_x_links = []
params_x_matter = []
params_y_matter = []
params_z_matter = []
params_zzz = []
params_zzzz = []
energies = []
gauss_callback = []

# Callback function to print the cost every 50 iterations
def callback(params):
    params_x_links.append(params[0])
    params_x_matter.append(params[n_links])
    params_y_matter.append(params[n_links+n_fermions-n_plaq])
    params_z_matter.append(params[n_links+2*(n_fermions-n_plaq)])
    params_zzz.append(params[2*n_links+2*(n_fermions-n_plaq)])
    params_zzzz.append(params[2*n_links+2*(n_fermions-n_plaq)+3*n_plaq+1])


plt.figure()
plt.plot(list(range(len(params_x_links))), params_x_links, color='blue', linestyle='solid', linewidth=2, label='x links')#+str(three_terms_n_layers)+"-layers")
plt.plot(list(range(len(params_x_matter))), params_x_matter, color='orange', linestyle='solid', linewidth=2, label='x matter')
plt.plot(list(range(len(params_z_matter))), params_z_matter, color='green', linestyle='solid', linewidth=2, label='z matter')
plt.plot(list(range(len(params_zzz))), params_zzz, color='red', linestyle='solid', linewidth=2, label='zzz')
plt.plot(list(range(len(params_zzzz))), params_zzzz, color='purple', linestyle='solid', linewidth=2, label='zzzz')
plt.xlabel("Steps", fontsize=15, color="#444444")
plt.ylabel("Parameters", fontsize=15, color="#444444")
plt.grid()
plt.show()'''

'''
fermion_inds = list(range(0, n_qubits))
    avoided_qubits = list(range(1, n_qubits, 2))
    fermion_inds = [q for q in fermion_inds if q not in avoided_qubits]
    avoided_qubits = [q*6+4 for q in range(0, n_plaq)]
    fermion_inds = [q for q in fermion_inds if q not in avoided_qubits]
    print(fermion_inds)
terms = get3bterms(n_plaq)

    for i, term in enumerate(terms):
        defects = [0] * n_qubits
        defects[term[0]] = 1
        defects[term[2]] = 1
        active_fermions = []
        for j, v in enumerate(defects):
            if j in fermion_inds:
                active_fermions.append(v)
        print(active_fermions)
        defects = [f*s for f,s in zip(active_fermions,ordering)]
        H_p = ladder_hamiltonian(n_plaq,0,0,0,0,1,defects)
        exp = expec(params, H_p)
        penalty_exp[i].append(exp)'''
