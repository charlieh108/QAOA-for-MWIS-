### Contour Plots

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

import graph_plot as gp

from qiskit.optimization.algorithms import CplexOptimizer
from docplex.mp.model import Model
from qiskit.optimization import QuadraticProgram
import numba
from numba import jit
import pickle


from matplotlib import ticker


############# Functions for converting Qiskit qubit ordering around #############

# Endianness conversion tools from https://github.com/Qiskit/qiskit-terra/issues/1148#issuecomment-438574708


############# Functions for converting Qiskit qubit ordering around #############

# Endianness conversion tools from https://github.com/Qiskit/qiskit-terra/issues/1148#issuecomment-438574708

def state_num2str(basis_state_as_num, nqubits):
    return '{0:b}'.format(basis_state_as_num).zfill(nqubits)

def state_str2num(basis_state_as_str):
    return int(basis_state_as_str, 2)

def state_reverse(basis_state_as_num, nqubits):
    basis_state_as_str = state_num2str(basis_state_as_num, nqubits)
    new_str = basis_state_as_str[::-1]
    return state_str2num(new_str)

def get_adjusted_state(state):
    nqubits = np.log2(state.shape[0])
    if nqubits % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    nqubits = int(nqubits)

    adjusted_state = np.zeros(2**nqubits, dtype=complex)
    for basis_state in range(2**nqubits):
         adjusted_state[state_reverse(basis_state, nqubits)] = state[basis_state]
    return adjusted_state


###############################################################################

######### Docplex Generation ###############

def quantum_operator_z_coefficients(Graph,model_name='Yo'):

    '''
    Generates the quanutm object to be passed to the optimal angles function and gets the coefficients
    for the Z terms. These are then

    '''
    number_of_nodes = Graph.number_of_nodes()
    qp = QuadraticProgram()

    qp.from_docplex(new_docplex_generator(Graph,'Model Name')) # Putting in Graph
    quantum_operator, offset = qp.to_ising()
    oplisted = quantum_operator.oplist
    Z_coeff_list =[]
    for i in range(number_of_nodes):
        coeff_i= oplisted[i].coeff
        Z_coeff_list.append(coeff_i)



    Z_coeff = np.array(Z_coeff_list)
    coeff = Z_coeff
   # print(type(quantum_operator))
   # print(quantum_operator.parameters," parameter list")
   # print(quantum_operator.primitive_strings()," prim list")
   # print(quantum_operator.oplist," details list")

    return coeff

def new_docplex_generator(G,model_name):
    '''

    Takes in a networkx graph with weighted nodes and creates the docplex model for the
    MWIS

    '''
    mdl = Model(model_name)

    n = G.number_of_nodes()
    x = mdl.binary_var_list('x_{}'.format(i) for i in range(n)) #creates list of variables for each node
    node_list = list(G.nodes())
    node_weights = G.nodes(data='node_weight')
    just_weights = [weight[1] for weight in node_weights] #gets just the node weight
    scale = max(just_weights) # used as J_i,j must be greater than weight of node; all node weights are scaled to below 0 and J_ij is put as 2


    edge_list = list(G.edges())

    node_weight_terms  = mdl.sum([x[i] * -1*(just_weights[i]/scale) for i in node_list])
    edge_indepedence_terms  = mdl.sum([2*x[i]*x[j] for (i,j) in edge_list])
    mdl.minimize(node_weight_terms + edge_indepedence_terms)  # does this need to be minimise ?
    #mdl.prettyprint()

    return mdl


##################### Circuit Generation & Statevector Conversions  ######################################








def get_exact_solution(Graph,model_name):

    qp = QuadraticProgram()
    qp.from_docplex(new_docplex_generator(Graph,model_name))
    cplex = CplexOptimizer()
    result = cplex.solve(qp)
    return result


def quantum_operator_for_graph(Graph,model_name):

    '''
    Generates the quanutm object to be passed to the optimal angles function

    '''
    qp = QuadraticProgram()
    qp.from_docplex(new_docplex_generator(Graph,model_name)) # Putting in Graph


    quantum_operator, offset = qp.to_ising()

    return quantum_operator
############################################  Jitted Energy Functions #############

##### Energy Calculation  ######

def state_to_ampl_counts(vec, eps=1e-15):
    """Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = '0{}b'.format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real**2+val.imag**2 > eps:
            counts[format(kk, str_format)] = val
    return counts


def compute_mwis_energy_sv(statevector, G):
    '''
    Computes objective value from an inputted dictionary of amplitudes and bit strings and Graph
    Optimised using function mapping from https://stackoverflow.com/questions/14372613/efficient-creation-of-numpy-arrays-from-list-comprehension-and-in-general/14372746#14372746
    '''
    counts = state_to_ampl_counts(statevector)

    bit_strings  = list(counts.keys())
    amplitudes = np.array(list(counts.values()))
    number_of_mwis_values = len(bit_strings)
    objective_values =np.fromiter((mwis_objective(bit_string,G) for bit_string in bit_strings),float,number_of_mwis_values)
    probabilities = np.abs(amplitudes)**2

    objective  = np.sum(probabilities * objective_values)

    return objective

def compute_mwis_energy_sv_bar_chart(statevector, G):
    '''
    Computes objective value from an inputted dictionary of amplitudes and bit strings and Graph
    Optimised using function mapping from https://stackoverflow.com/questions/14372613/efficient-creation-of-numpy-arrays-from-list-comprehension-and-in-general/14372746#14372746
    '''
    counts = state_to_ampl_counts(statevector)

    bit_strings  = list(counts.keys())
   # print("sv bitstrings\n",bit_strings)
    amplitudes = np.array(list(counts.values()))
    number_of_mwis_values = len(bit_strings)
    objective_values =np.fromiter((mwis_objective(bit_string,G) for bit_string in bit_strings),float,number_of_mwis_values)
    probabilities = np.abs(amplitudes)**2
   # print("statevector probs\n",list(probabilities))

    objective  = np.sum(probabilities * objective_values)

    return objective,probabilities,objective_values

def compute_mwis_energy_sv_success_prob(statevector, G):
    '''
    Computes objective value from an inputted dictionary of amplitudes and bit strings and Graph
    Optimised using function mapping from https://stackoverflow.com/questions/14372613/efficient-creation-of-numpy-arrays-from-list-comprehension-and-in-general/14372746#14372746
    '''
    counts = state_to_ampl_counts(statevector)

    bit_strings  = list(counts.keys())
    amplitudes = np.array(list(counts.values()))
    number_of_mwis_values = len(bit_strings)
    objective_values =np.fromiter((mwis_objective(bit_string,G) for bit_string in bit_strings),float,number_of_mwis_values)
    probabilities = np.abs(amplitudes)**2
    #print("statevector probs\n",list(probabilities))

    objective  = np.sum(probabilities * objective_values)

    return objective, probabilities, objective_values

####### Jitted Functions ############


def state_to_ampl_counts_jitted(vec, eps=1e-15):
    """Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = '0{}b'.format(qubit_dims)
    list_of_bitstrings = []
    amplitudes = []
    for kk in range(vec.shape[0]):
        val = vec[kk]

        if val.real**2+val.imag**2 > eps:
            bitstring = format(kk, str_format)
            array_of_x = np.array(list(bitstring), dtype=int)
            list_of_bitstrings.append(array_of_x)
            amplitudes.append(val)

    amplitudes = np.array(amplitudes)
    bitstrings = np.array(list_of_bitstrings)


    return amplitudes, bitstrings




def energy_wraparound(Graph,statevector):

    amplitudes,bitstrings = state_to_ampl_counts_jitted(statevector, eps=1e-15)

    node_weights = Graph.nodes(data='node_weight')  # this is how the node weight is stored in my graph attributes
    just_weights = np.array([weight[1] for weight in node_weights]) #gets just the node weight and converts it to a np array
    edges = np.array(Graph.edges())

    objective_value = compute_mwis_energy_sv_new(amplitudes,bitstrings,edges,just_weights)

    return objective_value



@jit(nopython = True)

def compute_mwis_energy_sv_new(amplitudes,bit_strings,edges,just_weights):
    '''
    Computes objective value from an inputted dictionary of amplitudes and bit strings and Graph
    Optimised using function mapping from https://stackoverflow.com/questions/14372613/efficient-creation-of-numpy-arrays-from-list-comprehension-and-in-general/14372746#14372746
    '''

    number_of_mwis_values = len(bit_strings)
    objective_values = np.empty(number_of_mwis_values)
    for i in range(len(objective_values)):

        objective_values[i] = mwis_objective_new(bit_strings[i],edges,just_weights)


    probabilities = np.abs(amplitudes)**2

    objective  = np.sum(probabilities * objective_values)
    #print("Final Objective Is",objective)
    return objective


@jit(nopython = True)

def compute_mwis_energy_sv_new_for_bar(amplitudes,bit_strings,edges,just_weights):
    '''
    Computes objective value from an inputted dictionary of amplitudes and bit strings and Graph
    Optimised using function mapping from https://stackoverflow.com/questions/14372613/efficient-creation-of-numpy-arrays-from-list-comprehension-and-in-general/14372746#14372746
    '''

    number_of_mwis_values = len(bit_strings)
    objective_values = np.empty(number_of_mwis_values)
    for i in range(len(objective_values)):

        objective_values[i] = mwis_objective_new(bit_strings[i],edges,just_weights)


    probabilities = np.abs(amplitudes)**2

    objective  = np.sum(probabilities * objective_values)
    #print("Final Objective Is",objective)
    return objective, probabilities,objective_values

def compute_mwis_energy_sv_bar_chart(statevector,graph):

    amplitudes,bitstrings = state_to_ampl_counts_jitted(statevector, eps=1e-15)

    node_weights = Graph.nodes(data='node_weight')  # this is how the node weight is stored in my graph attributes
    just_weights = np.array([weight[1] for weight in node_weights]) #gets just the node weight and converts it to a np array
    edges = np.array(Graph.edges())

    objective, probabilities,objective_values = compute_mwis_energy_sv_new_for_bar(amplitudes,bitstrings,edges,just_weights)

    return objective_value





@jit(nopython = True)
def mwis_objective_new(array_of_x,edges,just_weights):
    '''
    Takes in networkx graph  G and a bit string  x from the Qasm output and calculates the < psi | C | psi >
    Need to take note of the order of the bit string.
    '''
   # this takes the bit string 1001 to a numpy array for faster access
    # getting the maximum weight of nodes
    #node_weights = G.nodes(data='node_weight')  # this is how the node weight is stored in my graph attributes
   # just_weights = np.array([weight[1] for weight in node_weights]) #gets just the node weight and converts it to a np array
    scale = np.amax(just_weights) # gets the maximum weight so the node weights are
    scaled_weights = just_weights/scale  # used as J_i,j must be greater than weight of node; all node weights are scaled to below 0 and J_ij is put as 2

    objective = 0

    # independent set
    for edge_array in edges:
        i = edge_array[0]
        j = edge_array[1]

        if array_of_x[i] == 1 and  array_of_x[j]==1:  # interconnecting nodes are in the same set

            objective += 2



    weights_to_subtract = array_of_x * scaled_weights
    total_weight = np.sum(weights_to_subtract)

    objective = objective - total_weight

    return objective
################## Circuit Functions ################################


def append_x_term(qc, qubit_1, beta):
    qc.rx(2*beta, qubit_1)

def get_mixer_operator_circuit(graph, beta):
    N = graph.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for n in graph.nodes():
        append_x_term(qc, n, beta)
    return qc

def append_zz_term(qc,qubit_1,qubit_2,gamma):
    qc.cx(qubit_1,qubit_2)
    qc.rz(2*gamma, qubit_2)
    qc.cx(qubit_1,qubit_2)



def append_z_term(qc,qubit,gamma):
    qc.rz(-2*gamma,qubit)


def get_cost_operator_circuit(graph,coeff,gamma):


    n  = graph.number_of_nodes()

    qc = QuantumCircuit(n,n)

    for i,j in graph.edges():

        append_zz_term(qc,i,j,gamma*0.5)
    for i in graph.nodes():
        weight = coeff[i]
        append_z_term(qc,i,-gamma*weight)
    return qc



def get_qaoa_circuit_sv(G, coeff, beta, gamma):
    assert(len(beta) == len(gamma))
    p = len(beta) # infering number of QAOA steps from the parameters passed
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    # first, apply a layer of Hadamards
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        qc += get_cost_operator_circuit(G,coeff,gamma[i])
        qc += get_mixer_operator_circuit(G,beta[i])
    # no measurement in the end!
    return qc
###################################################





def calculate_energy(Graph,p,quantum_operator,angles, seed):

    '''

    Returns the state vector calculation for the objetive value energy for set angles and set p

    '''


    gamma = angles[:p] # First half gammas,
    beta = angles[p:]

    graph_nodes_number = Graph.number_of_nodes()
    coeff = quantum_operator_z_coefficients(Graph,model_name='Yo')
    quantum_circuit = get_qaoa_circuit_sv(Graph,coeff,beta,gamma)

    backend = Aer.get_backend('statevector_simulator')
    statevector = get_adjusted_state(execute(quantum_circuit, backend, seed_simulator=seed).result().get_statevector())

    objective_value = energy_wraparound(Graph,statevector)

    return objective_value



def z_function_expectation(Graph,gamma,beta,seed=1):

    '''

    Takes in mesh grid and hands over pair of angles to calcule_energy to build up contour plot
    '''
    quantum_operator = quantum_operator_for_graph(Graph," Energy Calculation")


    Objective_array = np.zeros((gamma.shape[0],gamma.shape[0])) # gets size of the matrix
    for i in range(int(gamma.shape[0])):
        for j in range(int(gamma.shape[0])):

            angles = [gamma[i,j], beta[i,j]]    # vectorises the show by doing two for loops to supply values
            angles_array = np.array(angles)
            energy = calculate_energy(Graph,1,quantum_operator,angles_array,seed)
            Objective_array[i,j] = energy
    return Objective_array



def contour_plot_expectation(Graph,gamma_range,beta_range,fidelity):

    gamma_range = np.arange(gamma_range[0],gamma_range[1],fidelity)
    beta_range = np.arange(beta_range[0],beta_range[1],fidelity)
    X,Y = np.meshgrid(gamma_range,beta_range)

    print("About to calculate...")

    Z = z_function_expectation(Graph,X,Y)
    x_pi =  X /np.pi
    y_pi = Y /np.pi
    graph_name = Graph.nodes[0]["graph_name"]
    fig, ax = plt.subplots(figsize=(16,12))

    contour = ax.pcolormesh(x_pi,y_pi,Z,cmap ='plasma',shading='nearest')
    colourbar = fig.colorbar(contour,ax=ax)
    colourbar.set_label(r'$ \langle\gamma,\beta|C|\gamma,\beta\rangle$ ')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax.grid(which='major',axis='both',color='w', linestyle='--', linewidth=0.75)
    #  ax.grid(which='minor',axis='both',color='w', linestyle='--', linewidth=0.5)


    #ScalarMappable.set_clim(vmin=None,vmax =6)
    ax.set_xlabel(r"$\gamma/\pi$")
    ax.set_ylabel(r"$\beta/\pi$")
    ax.set_title(graph_name)
    file_name = str(graph_name)

    pickle_6 = open((file_name +'.pkl'),'wb')
    pickle.dump(Z, pickle_6)
    pickle_6.close()
    fig.savefig(file_name+ ".png",dpi=450)



def z_function_expectation_success_prob(Graph,exact_solution,gamma,beta,seed=1):

    '''

    Returns an array of the success probability for various points. Each point has list of probabilities associated with an objective value. Where this objective value matches the correct
    solution to the problem, this probability is summed (may be more than one best solution) and returned as an array defined on the points given by the mesh grid
    '''
    quantum_operator = quantum_operator_for_graph(Graph," Energy Calculation")

    Objective_array = np.zeros((gamma.shape[0],gamma.shape[0]))
    for i in range(int(gamma.shape[0])):        # vectorises the show by doing two for loops to supply values
        for j in range(int(gamma.shape[0])):

            angles = [gamma[i,j], beta[i,j]]
            angles_array = np.array(angles)
            energy,probabilities,objective_values = calculate_energy_success_prob(Graph,1,quantum_operator,angles_array,seed)

            exact_solution_indices = np.array([solution_index for  solution_index,solution in np.ndenumerate(objective_values) if solution == exact_solution])  # gets indexes of places where the objective value in statevector matches exact solution
            success_probabilities = probabilities[exact_solution_indices]  #   https://stackoverflow.com/questions/25201438/python-how-to-get-values-of-an-array-at-certain-index-positions/25201506
            total_success_prob = np.sum(success_probabilities) # total probs added
            Objective_array[i,j] = total_success_prob
    return Objective_array




def calculate_energy_success_prob(Graph,p, quantum_operator,angles, seed):

    '''

    Returns the state vector calculation for the objetive value energy for set angles and set p

    '''
    graph_nodes_number = Graph.number_of_nodes()


    quantum_circuit = get_qaoa_circuit_sv(quantum_operator,p,angles,graph_nodes_number)

    backend = Aer.get_backend('statevector_simulator')
    statevector = get_adjusted_state(execute(quantum_circuit, backend, seed_simulator=seed).result().get_statevector())

    objective_value, probabilities,objective_values = compute_mwis_energy_sv_success_prob(statevector,Graph)

    return objective_value,probabilities,objective_values




def contour_plot_success_prob(Graph,gamma_range,beta_range,fidelity):

    gamma_range = np.arange(gamma_range[0],gamma_range[1],fidelity)
    beta_range = np.arange(beta_range[0],beta_range[1],fidelity)
    X,Y = np.meshgrid(gamma_range,beta_range)

    graph_name = Graph.nodes[0]["graph_name"]

    exact_result = get_exact_solution(Graph,"exact")
    exact_result_bitstring_array = list((exact_result.x).astype(int))
      #https://stackoverflow.com/questions/5618878/how-to-convert-list-to-string
    exact_result_bitstring = ("".join(str(bit) for bit in exact_result_bitstring_array)) # converting the exact cplex solution into a form that can be read by mwis_objective

    exact_energy = mwis_objective(exact_result_bitstring,Graph)

    success_array = z_function_expectation_success_prob(Graph,exact_energy,X,Y)

    fig_suc, ax_suc = plt.subplots()

    contour_suc = ax_suc.pcolormesh(X,Y,success_array)
    colourbarsuc = fig_suc.colorbar(contour_suc,ax=ax_suc)
    colourbarsuc.set_label(r'Probability of Success')
    ax_suc.set_xlabel(r"$\gamma$")
    ax_suc.set_ylabel(r"$\beta$")

    ax_suc.set_title(graph_name)
    file_name = str(graph_name) + " Success Prob.jpg"
    fig_suc.savefig(file_name)



# Step 1

#Choose Graph
#Examples
path7  = gp.weighted_path_graph(5,[1,1,1,1,1,1,1])
#path_weighted = gp.weighted_path_graph(5,[1,20,1,10,1,10,1])

erdos_weighted = gp.weighted_erdos_graph(5,0.5,seed = 4)
erdos_unweighted = gp.unweighted_erdos_graph(5,0.5,seed = 9)


# Can also draw the graph using the networkx
#gp.draw_unsorted_graph(erdos_weighted,gp.standard_colours(erdos_weighted))
#gp.draw_unsorted_graph(erdos_unweighted,gp.standard_colours(erdos_unweighted))


# Step 2 Do Mapping of Expectation (Choose limits for gamma and beta and the fidelity)

contour_plot_expectation(erdos_weighted,[0*np.pi,1*np.pi],[0*np.pi,1*np.pi],0.1)


# Step 3 Do Success prob graph using same method (Note, due to matplotlib drawing cannot do both functions in same call. )
# Examples 
#contour_plot_success_prob(path3,[-np.pi,np.pi],[-np.pi,np.pi],0.01)
#contour_plot_success_prob(path5,[-np.pi,np.pi],[-np.pi,np.pi],0.01)
#contour_plot_success_prob(path7,[-np.pi,np.pi],[-np.pi,np.pi],0.01)
#contour_plot_success_prob(gp.unweighted_ibm_5(),[0,2*np.pi],[0,2*np.pi],0.1)
