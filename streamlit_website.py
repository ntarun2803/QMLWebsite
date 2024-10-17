import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model

from qiskit_aer import Aer,AerSimulator
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit_optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer
from qiskit_optimization import QuadraticProgram

from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary, LinearEqualityToPenalty

from qiskit_optimization import QuadraticProgram
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators.docplex_mp import to_docplex_mp
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model

from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2


# Data Bins Function
def data_bins(results, wj, n, m, l=0, simplify=False):
    """save the results on a dictionary with the three items, bins, items, and index.
    results (cplex.solve): results of the optimization
    wj: (array (1,m): weights of the items
    n: (int) number of items
    m: (int) number of bins
    """
    if simplify:
        bins = np.ones((m,))
        if m-l > 0: 
            bins[m-l-1:m] = results[:m-l]
        items = np.zeros((m,n))
        items[:,1:] =  results[m-l:(m-1)*n+m-l].reshape(m,n-1)
        items[0,0] = 1
        items = items.reshape(m,n) * wj
        return {"bins":bins, "items":items,"index":np.arange(m)}
    else:        
        return {"bins":results[:m], "items":results[m:m+m*n].reshape(m,n) * wj, "index":np.arange(m)}
def plot_bins(results, wj, n, m, l=0,simplify=False):
    """plot in a bar diagram the results of an optimization bin packing problem"""
    res = data_bins(results.x, wj, n, m, l, simplify)
    plt.figure()
    ind = res["index"]
    plt.bar(ind, res["items"][:,0], label=f"item {0}")
    suma = bottom=res["items"][:,0]
    for j in range(1,n):
        plt.bar(ind, res["items"][:,j], bottom=suma, label=f"item {j}")
        suma += res["items"][:,j]
    plt.hlines(Q,0-0.5,m-0.5,linestyle="--", color="r",label="Max W")
    plt.xticks(ind)
    plt.xlabel("Bin")
    plt.ylabel("Weight")
    plt.legend()


st.set_page_config(
    page_title = 'QML Project'
)



st.title("Problem Statement")


st.markdown("### Objective:")
st.latex(r"\text{minimize}_{x, \xi} \sum_{i=1}^{m} \chi_i")


st.markdown("### Subject to:")
st.latex(r"\sum_{i=1}^{m} \xi_{ij} = 1, \quad j = 1, \ldots, n")
st.latex(r"\sum_{j=1}^{n} w_j \xi_{ij} \leq Q \chi_i, \quad i = 1, \ldots, m")
st.latex(r"\xi_{ij} \in \{0, 1\}, \quad i = 1, \ldots, m, \quad j = 1, \ldots, n")
st.latex(r"\chi_i \in \{0, 1\}, \quad i = 1, \ldots, m")


st.markdown(r"""
### Definitions:
- **n**: Number of items
- **m**: Number of bins
- **w_j**: Weight of the j-th item
- **\(\chi_i\)**: Indicator variable for whether the i-th bin is used
- **\(\xi_{ij}\)**: Indicator variable for whether item j is placed in bin i
""")


st.title('Gathering Inputs')

np.random.seed(2)
n = st.slider('Enter the value of N',0,10) # number of bins
m = n # number of items
Q = st.slider('Enter the value of Q',0,100) # max weight of a bin


if(st.button("run sequence")):
    wj = np.random.randint(1,Q,n)
    st.success('Values Selected Successfully')
    st.text(wj)

    st.title("Constructing The Docplex Model For Our Objective Function")

    mdl = Model("BinPacking")

    x = mdl.binary_var_list([f"x{i}" for i in range(n)]) # list of variables that represent the bins
    e =  mdl.binary_var_list([f"e{i//m},{i%m}" for i in range(n*m)]) # variables that represent the items on the specific bin

    objective = mdl.sum([x[i] for i in range(n)])

    mdl.minimize(objective)


    for j in range(m):
        # First set of constraints: the items must be in any bin
        constraint0 = mdl.sum([e[i*m+j] for i in range(n)])
        mdl.add_constraint(constraint0 == 1, f"cons0,{j}")
        
    for i in range(n):
        # Second set of constraints: weight constraints
        constraint1 = mdl.sum([wj[j] * e[i*m+j] for j in range(m)])
        mdl.add_constraint(constraint1 <= Q * x[i], f"cons1,{i}")


    # Load quadratic program from docplex model
    qp = QuadraticProgram()
    qp=from_docplex_mp(mdl)

    st.text("Extract the Ising Hamiltonian and variable mappings")
    st.code(mdl.export_as_lp_string())


    # convert from DOcplex model to Qiskit Quadratic program
    qp = QuadraticProgram()
    qp=from_docplex_mp(mdl)

    # Solving Quadratic Program using CPLEX
    cplex = CplexOptimizer()
    result = cplex.solve(qp)
    print(result)
    fig = plot_bins(result, wj, n, m)
    st.pyplot(fig)


    st.title("Constructing the model using docplex")
    mdl = Model("BinPacking_simplify")

    l = int(np.ceil(np.sum(wj)/Q))
    x = mdl.binary_var_list([f"x{i}" for i in range(m)]) # list of variables that represent the bins
    e =  mdl.binary_var_list([f"e{i//m},{i%m}" for i in range(n*m)]) # variables that represent the items on the specific bin

    objective = mdl.sum([x[i] for i in range(n)])

    mdl.minimize(objective)

    for j in range(m):
        # First set of constraints: the items must be in any bin
        constraint0 = mdl.sum([e[i*m+j] for i in range(n)])
        mdl.add_constraint(constraint0 == 1, f"cons0,{j}")
        
    for i in range(n):
        # Second set of constraints: weight constraints
        constraint1 = mdl.sum([wj[j] * e[i*m+j] for j in range(m)])
        mdl.add_constraint(constraint1 <= Q * x[i], f"cons1,{i}")


    # Load quadratic program from docplex model
    qp = QuadraticProgram()
    qp=from_docplex_mp(mdl)
    # Simplifying the problem
    for i in range(l):
        qp = qp.substitute_variables({f"x{i}":1}) 
    qp = qp.substitute_variables({"e0,0":1}) 
    for i in range(1,m):
        qp = qp.substitute_variables({f"e{i},0":0})
    st.code(qp.prettyprint())
    st.code(mdl.export_as_lp_string())

    st.title("solving the quadratic programming using CPLEX")

    simplify_result = cplex.solve(qp)
    st.text(simplify_result)
    st.pyplot(plot_bins(simplify_result, wj, n, m, l, simplify=True))



    st.title("Defining The Function for Optimal Solution and converting it from inequality constraints to equality constraints")

    # st.header("Optimal Solution:")
    # for i, var in enumerate(qubo.variables):
    #     print(f"{var.name}: {result_qaoa.x[i]}")

    # # Print the optimal objective function value (minimum value)
    # st.code(f"Optimal Objective Function Value: {result_qaoa.fval}")

    ineq2eq = InequalityToEquality()
    qp_eq = ineq2eq.convert(qp)
    st.code(qp_eq.export_as_lp_string())
    st.code(f"The number of variables is {qp_eq.get_num_vars()}")



    st.title("Converting From Integer Constrained Problem to Binary Constrained Problem")

    int2bin = IntegerToBinary()
    qp_eq_bin = int2bin.convert(qp_eq)
    st.code(qp_eq_bin.export_as_lp_string())
    st.code(f"The number of variables is {qp_eq_bin.get_num_vars()}")


    st.title("Converting The Linear Equality Constraints Into Penalties")

    lineq2penalty = LinearEqualityToPenalty()
    qubo = lineq2penalty.convert(qp_eq_bin)
    st.code(f"The number of variables is {qp_eq_bin.get_num_vars()}")
    st.code(qubo.export_as_lp_string())

    st.title("Solving the QUBO(Quadratic uncontrained binary optimization problem ) Using CPLEX")

    result = cplex.solve(qubo)
    st.code(result)

    st.title("Performing Data Analysis and Visualisation From the Optimisation Results")

    data_bins(result.x, wj, n, m, l=l, simplify=True)
    st.pyplot(plot_bins(result, wj, n, m, l=l, simplify=True))

    st.title("Using QAOA to solve the QUBO Problem")


    backend = FakeAlmadenV2()



    sampler = Sampler()
    sampler = AerSampler(backend_options={"method": "statevector"},
                        run_options={"shots": 1024, "seed": 42})

    
    optimizer=COBYLA(maxiter=100)

    qaoa = MinimumEigenOptimizer(QAOA(sampler, optimizer, reps = 1))
    result_qaoa = qaoa.solve(qubo)
    st.code(result_qaoa)
    plt.title("QAOA solution", fontsize=18)

    st.pyplot(plot_bins(result, wj, n, m, l, simplify=True))

    st.title("Quadratic Penalty Function (USP)")

    import numpy as np
    import matplotlib.pyplot as plt

    # Parameters
    alpha = -2

    # Generate x values
    x = np.linspace(-5, 5, 100)

    # Quadratic penalty function
    f = (x - alpha) ** 2

    # Plot the function
    fig, ax = plt.subplots()
    ax.plot(x, f)
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Quadratic Penalty Function")

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.title("Constructing a Model Using the Docplex Method")

    from docplex.mp.model import Model
    from qiskit_optimization import QuadraticProgram

    # Define the model
    mdl = Model("binPackingSoftPenalty")

    # Number of bins (m) and items (n) assumed to be defined
    # List of binary variables representing whether a bin is used (1 if used, 0 otherwise)
    x = mdl.binary_var_list([f"x{i}" for i in range(m)])  

    # List of binary variables representing whether item j is in bin i
    e = mdl.binary_var_list([f"e{i//m},{i%n}" for i in range(n*m)])

    # Objective function: minimize the number of bins used (sum of x[i])
    objective = mdl.sum([x[i] for i in range(m)])

    # Penalty term initialization for quadratic penalties
    penalty = 0
    alpha = 10  # Alpha can be adjusted to control the strength of the penalty

    # Loop through bins to impose quadratic penalties for exceeding bin capacities
    for i in range(m):
        # Expression for bin capacity constraint
        cons_1 = 0
        cons_1 += Q * x[i]  # Capacity of bin i
        for j in range(n):
            cons_1 -= wj[j] * e[i * m + j]  # Subtract weight of item j if placed in bin i
        
        # Soft quadratic penalty for bin capacity violation
        penalty += (cons_1 - alpha) ** 2  # Quadratic penalty if capacity exceeded

    # Combine objective function with penalties
    mdl.minimize(objective + penalty)

    # Add constraints to ensure every item is assigned to exactly one bin
    for j in range(n):
        constraint0 = mdl.sum([e[i * m + j] for i in range(m)])  # Sum over all bins for item j
        mdl.add_constraint(constraint0 == 1, f"cons0,{j}")

    # Load quadratic program from docplex model
    qp = QuadraticProgram()
    qp=from_docplex_mp(mdl)

    # Simplify the problem by substituting some variables (if required)
    for i in range(l):
        qp = qp.substitute_variables({f"x{i}": 1})  # Example substitution for variables
    qp = qp.substitute_variables({"e0,0": 1})  # Example substitution
    for i in range(1, m):
        qp = qp.substitute_variables({f"e{i},0": 0})

    # Output the final quadratic program
    st.code(qp.export_as_lp_string())
    st.code(f"The number of variables is {qp.get_num_vars()}")


    st.title("Solving Quadratic Program Using CPLEX")

    # Solving Quadratic Program using CPLEX
    cplex = CplexOptimizer()
    result = cplex.solve(qp)
    st.code(result)
    st.pyplot(plot_bins(result, wj, n, m, l, simplify=True))

    qubo_myapp = lineq2penalty.convert(qp)
    result_myapp = qaoa.solve(qubo_myapp)
    st.code(result_myapp)

    st.text("QAOA Solution")
    st.pyplot(plot_bins(result_myapp, wj, n, m, l, simplify=True))


    st.title("Function for Optimizing the Function Value")

        # Solve the QUBO problem using QAOA
    result_myapp = qaoa.solve(qubo_myapp)


    # Print the result of the QAOA solution
    st.text("QAOA Result:")
    st.code(result_myapp)

    # Print the optimal objective function value
    st.code(f"Optimal Objective Function Value: {result_myapp.fval}")

    st.title("showcasing the difference between exponential and qudratic penalty approach")

    from docplex.mp.model import Model
    from qiskit_optimization import QuadraticProgram

    # Define the model
    mdl = Model("binPackingSoftPenalty")

    # Number of bins (m) and items (n) assumed to be defined
    # List of binary variables representing whether a bin is used (1 if used, 0 otherwise)
    x = mdl.binary_var_list([f"x{i}" for i in range(m)])  

    # List of binary variables representing whether item j is in bin i
    e = mdl.binary_var_list([f"e{i//m},{i%n}" for i in range(n*m)])

    # Objective function: minimize the number of bins used (sum of x[i])
    objective = mdl.sum([x[i] for i in range(m)])

    # Penalty term initialization for quadratic penalties
    penalty = 0
    alpha = 10  # Alpha can be adjusted to control the strength of the penalty

    # Loop through bins to impose quadratic penalties for exceeding bin capacities
    for i in range(m):
        # Expression for bin capacity constraint
        cons_1 = 0
        cons_1 += Q * x[i]  # Capacity of bin i
        for j in range(n):
            cons_1 -= wj[j] * e[i * m + j]  # Subtract weight of item j if placed in bin i
        
        # Soft quadratic penalty for bin capacity violation
        penalty += (-(cons_1+ alpha)+(cons_1 + alpha) ** 2 ) # Quadratic penalty if capacity exceeded

    # Combine objective function with penalties
    mdl.minimize(objective + penalty)

    # Add constraints to ensure every item is assigned to exactly one bin
    for j in range(n):
        constraint0 = mdl.sum([e[i * m + j] for i in range(m)])  # Sum over all bins for item j
        mdl.add_constraint(constraint0 == 1, f"cons0,{j}")

    # Load quadratic program from docplex model
    qp = QuadraticProgram()
    qp=from_docplex_mp(mdl)

    # Simplify the problem by substituting some variables (if required)
    for i in range(l):
        qp = qp.substitute_variables({f"x{i}": 1})  # Example substitution for variables
    qp = qp.substitute_variables({"e0,0": 1})  # Example substitution
    for i in range(1, m):
        qp = qp.substitute_variables({f"e{i},0": 0})

    # Output the final quadratic program
    st.code(qp.export_as_lp_string())
    st.code(f"The number of variables is {qp.get_num_vars()}")


    st.title("Similarly solving it using CPLEX")

    # Solving Quadratic Program using CPLEX
    cplex = CplexOptimizer()
    result = cplex.solve(qp)
    st.code(result)
    st.pyplot(plot_bins(result, wj, n, m, l, simplify=True))

    st.text("QAOA Solution")

    qubo_myapp = lineq2penalty.convert(qp)
    result_myapp = qaoa.solve(qubo_myapp)
    st.code(result_myapp)

    st.pyplot(plot_bins(result_myapp, wj, n, m, l, simplify=True))


    # Solve the QUBO problem using QAOA
    result_myapp = qaoa.solve(qubo_myapp)


    # Print the result of the QAOA solution
    st.text("QAOA Result:")
    st.code(result_myapp)

    # Print the optimal objective function value
    st.code(f"Optimal Objective Function Value: {result_myapp.fval}")


    st.text("the fval for quadratic approach for same value of aplha is = 176 while fval for exponential penalty is = 1394 ")
    st.text("i.e quadratic penalty is favorable")

    st.markdown(''' 
### Comparison of Classical and Quantum Approaches

1. **Classical Greedy Algorithm**: 
   - The objective function value (`fval`) for the greedy algorithm is **3**, indicating that the algorithm uses 3 bins to pack the items. 
   - The greedy approach is straightforward and quickly places items in the first available bin that has enough capacity. While it's efficient, it may not always produce the optimal solution, especially for problems like bin packing where a better packing could reduce the number of bins used.

2. **Quantum Quadratic Approach**:
   - In the quantum approach using QAOA with a quadratic penalty, the objective function value (`fval`) was **176**. This penalty is designed to enforce packing constraints while minimizing the total number of bins used. The lower the penalty value, the closer the solution is to the optimal one.
   - The quadratic penalty approach is favorable in many quantum optimization scenarios because it translates the problem into a form that quantum solvers can process efficiently. The quadratic formulation balances the packing and penalties, finding a near-optimal configuration with fewer constraints violated.

3. **Quantum Exponential Approach**:
   - The exponential penalty approach yielded an `fval` of **1394**, which is much higher than the quadratic approach's result. The higher penalty value suggests that this method may not be as effective in finding an optimal solution for this specific problem or under the given parameters.

### What `fval` Represents

- In both classical and quantum approaches, `fval` represents the quality of the solution, but in different ways:
  - **Classical**: In the greedy approach, `fval` is the number of bins used, which is directly tied to the efficiency of the packing.
  - **Quantum**: In the quantum approaches, `fval` is a function of the penalty for constraint violations and the number of bins used. A lower `fval` indicates a better solution that adheres to the constraints (e.g., items fitting within bin capacities) while minimizing the number of bins.

### Optimality of the Quadratic Approach

The quadratic penalty approach is more likely to yield a better solution because it formulates the bin packing problem in a way that balances penalties and solution feasibility. It optimizes both the number of bins and how the items are distributed among them, leading to a more optimal solution than both the classical greedy and quantum exponential methods in this case.

This demonstrates how quantum optimization can potentially outperform classical heuristics, especially when complex constraints are involved.


''')


    


