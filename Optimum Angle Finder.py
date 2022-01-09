
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.optimize import minimize
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.aqua.algorithms import NumPyEigensolver
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import op_converter
from qiskit.aqua.operators import WeightedPauliOperator
import graph_plot as gp

from qiskit.optimization.algorithms import CplexOptimizer
from docplex.mp.model import Model
from qiskit.optimization import QuadraticProgram
import numba
from numba import jit
import pickle
from joblib import Parallel, delayed

######################


# Code explained at bottom 



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

######### Starting Angle Generation ##############

def random_intial_values(gamma_interval, beta_interval, p,seed):
    '''
    Generate a uniform random array of betas and gammas length p for input into the angle optimiser
    Creates two arrays for each angle full of the min and max possible angles.
    numpy function that chooses random angle between these two values then acts
    '''
    lower_beta = beta_interval[0]
    higher_beta = beta_interval[1]  # gets the different values

    lower_gamma = gamma_interval[0]
    higher_gamma = gamma_interval[1]

    lower_beta_array= np.full(p,lower_beta) # creates the lower arrays
    lower_gamma_array = np.full(p,lower_gamma)
    lower_array = np.hstack((lower_gamma_array,lower_beta_array))

    higher_beta_array =np.full(p,higher_beta)
    higher_gamma_array = np.full(p,higher_gamma)
    higher_array = np.hstack((higher_gamma_array,higher_beta_array))
    np.random.seed(seed)
    random_angles = np.random.uniform(lower_array,higher_array, 2*p)

    return random_angles

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
    mdl.minimize(node_weight_terms + edge_indepedence_terms)
    #mdl.prettyprint()

    return mdl


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
    amplitudes = np.array(list(counts.values()))
    number_of_mwis_values = len(bit_strings)
    objective_values =np.fromiter((mwis_objective(bit_string,G) for bit_string in bit_strings),float,number_of_mwis_values)
    probabilities = np.abs(amplitudes)**2

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
    return objective

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





####### Circuit Generation  and Objective ##########

def get_black_box_objective_sv(G,p,coeff):
    backend = Aer.get_backend('statevector_simulator')
    def f(theta):
        gamma = theta[:p] # First half gammas,
        beta = theta[p:]  # Second half betas
        qc = get_qaoa_circuit_sv(G,coeff,beta, gamma)
        sv = execute(qc, backend).result().get_statevector()
        adjusted_statevector = get_adjusted_state(sv)
        # return the energy
        return energy_wraparound(G,adjusted_statevector)
    return f


def mwis_objective(x,G):
    '''
    Takes in networkx graph  G and a bit string  x from the Qasm output and calculates the < psi | C | psi >
    Need to take note of the order of the bit string.
    '''
    array_of_x = np.array(list(x), dtype=int) # this takes the bit string 1001 to a numpy array for faster access
    # getting the maximum weight of nodes
    node_weights = G.nodes(data='node_weight')  # this is how the node weight is stored in my graph attributes
    just_weights = np.array([weight[1] for weight in node_weights]) #gets just the node weight and converts it to a np array
    scale = np.amax(just_weights) # gets the maximum weight so the node weights are
    scaled_weights = just_weights/scale  # used as J_i,j must be greater than weight of node; all node weights are scaled to below 0 and J_ij is put as 2

    objective = 0
    for i,j in G.edges():  # independent set
        if array_of_x[i] == 1 and  array_of_x[j]==1:  # interconnecting nodes are in the same set

            #J_ij = np.min([scaled_weights[i],scaled_weights[j]])
           # print([just_weights[i],just_weights[j]])
            #print(J_ij)
            objective += 2


    for i in G.nodes():
        if array_of_x[i] == 1:
            objective -=scaled_weights[i]
            #objective -=just_weights[i]

    return objective

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


############### Testing Functions ###################

def get_exact_solution(Graph,model_name ='Exact Solution (s)'):

    qp = QuadraticProgram()
    qp.from_docplex(new_docplex_generator(Graph,model_name))
    cplex = CplexOptimizer()
    result = cplex.solve(qp)
    #print(result)

    return result.x

def get_exact_energy(Graph,model_name ='Exact Solution (s)'):

    qp = QuadraticProgram()
    qp.from_docplex(new_docplex_generator(Graph,model_name))
    cplex = CplexOptimizer()
    result = cplex.solve(qp)

    return result.fval



def string_amplitude_bar_chart(G,statevector,angles,solution_list = None):


    graph_name = G.nodes[0]["graph_name"]
    angle_string = str(angles)
    counts = state_to_ampl_counts(statevector)
    #print("counts",counts)
    bit_strings  = list(counts.keys())
    #print("bitstrings",bit_strings)
    amplitudes = np.array(list(counts.values()))
    #print("ampltides",amplitudes)
    probabilities = np.abs(amplitudes)**2
    indices_of_solution_list = [int(s,2) for s in solution_list] # takes the list of binary number strings into actual numbers ready to be imposed on https://note.nkmk.me/en/python-list-str-num-conversion/
    plt.bar(bit_strings,probabilities, align='center')
    plt.bar(indices_of_solution_list,probabilities[indices_of_solution_list],color='g')
    plt.xticks(range(len(bit_strings)),bit_strings,rotation = 90)
    ordered_probs_indices = np.argsort(probabilities)[::-1]
    ordered_probs = probabilities[ordered_probs_indices]
    ordered_bitstrings = (np.array(bit_strings))[ordered_probs_indices]
    objective_values =np.fromiter((mwis_objective(bit_string,G) for bit_string in ordered_bitstrings),float,len(ordered_bitstrings))

    #print(ordered_probs)
    print(ordered_bitstrings)
    print(objective_values)
    plt.title('Bit String Amplitude ' +graph_name  + "\n at " + angle_string )
    plt.show()


def objective_bar_chart(G,statevector,angles):

    objective,probabilities,objective_values = compute_mwis_energy_sv_bar_chart(statevector,G)
    graph_name = G.nodes[0]["graph_name"]
    angle_string = str(angles)

    plt.bar(objective_values,probabilities)
    plt.title('Objective Bar Chart for ' +graph_name + "\n at " + angle_string )
    plt.show()



def convert_solution_array_to_string(array):
    '''

    Turns numpy solution array into string
    '''
    #print("array b",array)
    array  = array.astype(int)
    #print("array a",array)
    list_of_numbers = list(array)

    # Converting integer list to string list
    s = [str(i) for i in list_of_numbers]

    # Join list items using join()
    res = str(int("".join(s)))
    #print(res)

    return(res)


def testing_function(Graph,angles,p):

        '''

        Put in a graph and angles, runs the circuit and get out plot etc and objective function values

        '''

        gamma = angles[:p] # First half gammas,
        beta = angles[p:]

        exact_solution = convert_solution_array_to_string(get_exact_solution(Graph))


        graph_nodes_number = Graph.number_of_nodes()

        seed =1
        coeff = quantum_operator_z_coefficients(Graph,model_name='Yo')
        quantum_circuit = get_qaoa_circuit_sv(Graph,coeff,beta,gamma)
        backend = Aer.get_backend('statevector_simulator')
        statevector = get_adjusted_state(execute(quantum_circuit, backend, seed_simulator=seed).result().get_statevector())
        #print("The Counts are ",amplitude_dictionary)
        string_amplitude_bar_chart(Graph,statevector,angles,[exact_solution,])
        objective_bar_chart(Graph,statevector,angles)
        objective_value = compute_mwis_energy_sv(statevector,Graph)
        print("Objective Value", objective_value)


########## Getting Optimal Angles #########


def get_optimal_angles(G,p,coeff,objective_function,initial_gamma_range,initial_beta_range,seed):
    '''
    This performs the classical-quantum interchange, improving the values of beta and gamma by reducing the value of
    < beta, gamma | C | beta, gamma >. Returns the best angles found and the objective value this refers to. Bounds on the minimiser are set as the starting points
    '''
    initial_starting_points = random_intial_values((np.array(initial_gamma_range)),(np.array(initial_beta_range)),p,seed)

    optimiser_function =  minimize(objective_function, initial_starting_points, method='COBYLA', options={'maxiter':1500})
    best_angles = optimiser_function.x
    objective_value = optimiser_function.fun
    return best_angles,objective_value




###################

def calculate_energy_success_prob(Graph,p,exact_solution, coeff,angles, angle_seed):

    '''

    takes the best angles and runs the circuit again to get the total chance of getting the best possible energy
    '''
    gamma = angles[:p] # First half gammas,
    beta = angles[p:]

    coeff = quantum_operator_z_coefficients(Graph,model_name='Yo')
    quantum_circuit = get_qaoa_circuit_sv(Graph,coeff,beta,gamma)
    backend = Aer.get_backend('statevector_simulator')
    statevector = get_adjusted_state(execute(quantum_circuit, backend, seed_simulator=angle_seed).result().get_statevector())
    objective_value, probabilities,objective_values = compute_mwis_energy_sv_success_prob(statevector,Graph)


    exact_solution_indices = np.array([solution_index for  solution_index,solution in np.ndenumerate(objective_values) if solution == exact_solution])  # gets indexes of places where the objective value in statevector matches exact solution
    success_probabilities = probabilities[exact_solution_indices]  #   https://stackoverflow.com/questions/25201438/python-how-to-get-values-of-an-array-at-certain-index-positions/25201506
    total_success_prob = np.sum(success_probabilities)

    return total_success_prob


def get_best_results(list_of_results):

    '''
    data is in the form
    [angle_seed,best_objective_value_for_seed,success_prob,best_angles_for_seed]

    gets the index of the largest success prob and the largest objective Value

    '''


    success_values = []
    objective_values = []

    for result in range(len(list_of_results)):

        objective_value = list_of_results[result][1]  # adds each objective value to new list of just these, where min can be found and averaged yo
        objective_values.append(objective_value)
        success_prob =list_of_results[result][2]
        success_values.append(success_prob)

    objective_values_array = np.array(objective_values)
    success_values_array = np.array(success_values)


    minimum_obj_values_indexes = np.where(objective_values_array == objective_values_array.min())
    best_success_values_indexes = np.where(success_values_array == success_values_array.max())

    array_of_results = np.array(list_of_results,dtype = object)
    best_objective_value_data_lists = list(array_of_results[minimum_obj_values_indexes])
    best_success_value_data_lists = list(array_of_results[best_success_values_indexes])



    list_of_best = [best_objective_value_data_lists,best_success_value_data_lists]

    return list_of_best






########################################## `

#Getting the 2p optimum angles for a set of random graphs from  to discern general patterns. Used to create figures 3 & 4 in diss.

# Step One

#Set p, number of graphs to try and the number of starting angle positions to then optimise from

number_of_angle_points = 10  # ~1000 is the minimum needed
number_of_graphs = 8
p =6  #### Number of stages


#Step Two: Choose size of graph and whether to solve MIS or MWIS
#In this function, choose the size of graph ( no. of nodes) you want to operate on and whether you want to solve the MIS problem or MWIS (weighted or unweighted nodes)


def find_best_angles_for_graph(graph_seed):
    print("14:On graph ",graph_seed)
    #graph = gp.weighted_erdos_graph(10,0.4,graph_seed)  # Weighted version
    graph = gp.unweighted_erdos_graph(10,0.4,graph_seed)  ####### Size of Graph Here
    graph_coefficients = quantum_operator_z_coefficients(graph,'Yo')
    exact_energy =get_exact_energy(graph)
    angle_starting_seed = np.arange(1,number_of_angle_points,1)

    objective_function= get_black_box_objective_sv(graph,p,graph_coefficients)
    list_of_results = []
    for angle_seed in angle_starting_seed:
        print("On Angle Seed",angle_seed)
        best_angles_for_seed, best_objective_value_for_seed = get_optimal_angles(graph,p,graph_coefficients,objective_function,[0,np.pi],[0,np.pi],angle_seed)
        success_prob = calculate_energy_success_prob(graph,p,exact_energy, graph_coefficients,best_angles_for_seed,angle_seed)
        angle_seed_data_list = [angle_seed,best_objective_value_for_seed,success_prob,best_angles_for_seed]
        list_of_results.append(angle_seed_data_list)

    list_of_best =  get_best_results(list_of_results)

    full_results = [graph_seed,exact_energy,list_of_best,list_of_results]

    return full_results




#### Step 3

# Choose Filename to store  pickled angle results  under

filename = 'unweighted_erdos_graph_N_9_p_7.pkl' # Typical form I used

# Choose number of cores that joblib uses

number_of_cores = -2 # -2 uses all but two cores


###### Get filename
list_of_results_every_graph = []
list_of_all_graph_results = Parallel(n_jobs=-2)(delayed(find_best_angles_for_graph)(i) for i in range(1,number_of_graphs+1))

with open(filename,'wb') as f:

    pickle.dump(list_of_all_graph_results, f)
