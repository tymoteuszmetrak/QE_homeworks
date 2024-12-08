

using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, NLsolve, Optim, Roots, Calculus, DataFrames
#Combined Solutions
# CC: Jan Hincz, Tymoteusz Metrak, Wojciech Szymczak
# QE 2024

# Problem 1 
function naive_optimization(fun, guess, α; ϵ = 1e-6, maxiter=1000)
    # The guess will be the first x input to the algorithm
    x_old = guess

    # Setting up a vector with all the x values
    x_vector = []
    x_vector = push!(x_vector, x_old)

    # Defining function g
    function g(x)
        return fun(x)+x
    end

    # Applying modified version of the algorithm
    function mod_algorithm(x)
        x = (1-α)*g(x)+α*x
        return x
    end

    # First calculation of the root
    x_new = mod_algorithm(x_old)

    # Add x value to the vector
    x_vector = push!(x_vector, x_new)

    # Create a vector of residuals
    residual = abs(x_new-x_old)
    residuals = []
    residuals = push!(residuals, residual)

    # Calculate the parameter for comparision with the epsilon
    closeness = residual/(1+abs(x_old))

    # Number of iterations
    count = 1

    # Introduce flag integer: 1 - the algorithm was stopped, 0 - the solution was found
    flag = 0

    # The loop that runs as long as the solution is further from the truth then epsilon
    while closeness>=ϵ
        x_old = x_new
        x_new = mod_algorithm(x_old)
        residual = abs(x_new-x_old)
        residuals = push!(residuals, residual)
        closeness = residual/(1+abs(x_old))
        count +=1
        x_vector = push!(x_vector, x_new)

        # Breaking the loop if maximum iterations are reached
        if count >= maxiter
            println("Maximum number of iternations was reached")
            # Flag set to 1, if the result was not found
            flag = 1
            # Solution is set to not a number, if it wasn't found
            x_new = NaN
            break
        end
    end

    # Value of the solution is calculated below
    if isnan(x_new)
        value = NaN
        difference =NaN
    else
        value = fun(x_new)
        # Calculating the absolute difference between x and g(x)
        difference = abs(x_new - g(x_new))
    end
    if count < maxiter
        println("The solutuion was found after $count iterations")
    end
        return (Flag = flag, 
                root = x_new, 
                value = value, 
                absolute_difference = difference,
                vector_of_all_x = x_vector, 
                vector_of_all_residuals = residuals)
end



### Testing the function ###
function f(x)
    return (x+1)^(1/3)-x
end

# Let's start with 1 starting point and alpha=0
naive_optimization(f, 1, 0)

# We didn't obtain any reasonable answer. We set alpha to 0, so the algorit

#  Now we'll do some experimenting with alpha and starting point
# Let's increase the parameter responsible for dampening
naive_optimization(f, 1, 0.5)
# It resulted only with more iterationa (23 instead of 9). Let's increase it more
naive_optimization(f, 1, 0.99)
# The algorithm works  well, it still finds the solution.
# However, it took as much as 867 iterations!

# Now let's tamper the starting point
# The function is in real number, so the smallest guess may be -1
# Let's see that on a plot

plot(f, -2,5, label = "function f(x)")

# We'll start with -1, instead of 1
naive_optimization(f, -1, 0)
# It converged from -1 and needed 11 iterations
# Let's start from 0 instead
naive_optimization(f, 0, 0)
# The algorithm worked as well
# Will it find a soltion if we move more to right on a plot?
naive_optimization(f, 10, 0)
# we found it and only after 10 iterations!

# We saw that changing the alpha significantly influences the optimization (number of iterations)
# However the initial guess seems to have lower influence on the results and the speed of getting there.

# Below we will test wheteher the solution will be found
# for a different function
function h(x)
    return x^3-x-1
end
naive_optimization(h, 1,0)
# Unfortunately, with this form of function the algorithm does not work

# Theoretically, now algorithm "coverges" to the root.
naive_optimization(h, 1.3247,1)

# It is however just hacking the algorithm.
# This algorithm is oversimplified and naive 
# and works well only for a very limited number of function


# For example, it should work for the following:
function m(x)
    return (x)^(1/5)-x
end
plot(m)
naive_optimization(m,0.5,0.5)

##### Problem 2

# At first we calculate an exact solution
function exact(α, β)
    x = [1, 1, 1, 1, 1]
    return(x)
end

# The α and β may take any value, they are not needed for calculations
exact(0,0)


# Then we proceed to a function, that present both exact solution 
# and the one obtained with backslash operator
function solver(α, β)
    # The exact solution
    x_exact = [1, 1, 1, 1, 1]
    
    # We define matrix A and matrix b
    A = [1 0 0 0 0; -1 1 0 0 0; 0 -1 1 0 0; (α-β) 0 -1 1 0; β 0 0 -1 1]'
    b = [α, 0, 0, 0, 1]

    # Calculating the vector x with backslash operator
    x_backslash = A\b

    # Relative residual and condition number
    relative_residual = norm(A*x_backslash - b)
    condition_number = cond(A)

    return(x_exact, x_backslash, relative_residual, condition_number)
end

# Let's look at selected outcomes depending on β
result = solver(0.1, 1)
result = solver(0.1, 10)
result = solver(0.1, 100)
result = solver(0.1, 1000)
result = solver(0.1, 1000000000000)


# In order to create a table that shows x1  the condition number, and the relative residual
# we build  a function that does so for a given number of β.
# We supply a maximum power, to which we take 10 and calculate β.


function create_table(power)
    # Powers of 10
    n = []
    # Calculated β
    betas = []
    # Exact values of x_1 always equal 1
    x_1_exact = []
    # x_1 values calculated using backslash operator
    x_1_values = []
    # Condition number
    cond_num_values = []
    # relative residual
    rel_resid_values = []

    for i in 0:power
        n = push!(n, i)
        betas = push!(betas, 10^i)
        x_1_exact = push!(x_1_exact, 1)
        result_now = solver(0.1, 10^i)
        x_1_values = push!(x_1_values, result_now[2][1])
        cond_num_values = push!(cond_num_values, result_now[4])
        rel_resid_values = push!(rel_resid_values, result_now[3])
    end
    data = hcat(n, betas, x_1_exact,x_1_values, cond_num_values, rel_resid_values)
    header = (["Power of 10", "β", "Exact x_1", "x_1 using backslash operator", "Condition number", "Relative residual"])
    pretty_table(data;
        header = header,
        header_crayon = crayon"yellow bold",
        tf = tf_unicode_rounded
        )
end

# Now, we create a table!
create_table(12)

###Problem 3: Internal rate of return 
function internal_rate(C::Vector{Float64})
    function NPV(r::Float64, C::Vector{Float64})
        discount_vector = [] 
        for i in 1:length(C)
            discount_factor = (1.0+r)^-(i-1.0)
            push!(discount_vector, discount_factor)
        end
        return discount_vector'*C
    end
    wrapped_NPV(r) = [NPV(r[1], C)] #r as a vector to accommodate nlsolve algorithm
    initial_guess = [0.1]
    IRR = nlsolve(wrapped_NPV, initial_guess; ftol=1e-14, show_trace=true);
    if IRR.f_converged == false || IRR.zero[1] <= -1.0
        println("WARNING: the solver did not find a valid IRR")
    else
        return IRR.zero[1]
    end
end

internal_rate([-3.0,4.5,4.5,4.5,4.5,4.5,4.5])

internal_rate([-5,0,0,2.5,5])

internal_rate([-1,1.1])

internal_rate([1,1.1])
internal_rate([-1,-1.1])
internal_rate([1.0,1.0])

#### PROBLEM 4 
# 4.1. Write a function that takes \alpha (a) \sigma (s) 
# and x1, x2 as arguments and creates a contour plot of the 
# production function for x1, x2
function plot_production(x, a, s)
    # Define the grid
    grid_x1 = 0:0.1:x[1]
    grid_x2 = 0:0.1:x[2]
    x1, x2 = [grid_x1, grid_x2]

    # Define the production function with a conditional for s
    function production(x, a, s) 
        if s == 1  
            return((x[1].^a) .* (x[2].^(1-a))).^0.5
        else
            return (a .* x[1].^((s-1)/s) + (1-a) .* x[2].^((s-1)/s)).^(s/(s-1))
        end
    end

    # Plot     
    gr_plot = plot(grid_x1, grid_x2, (x1, x2) -> production([x1,x2], a, s), st=:contour,color=:turbo, levels = 20,clabels=true, cbar=false, lw=1)
    return gr_plot
end

# Test 4.1 
plot_production([10,10], 0.5, 1) # arbitrary set 10, 10 as maximum inputs 

# 4.2  Create the above plot for α = 0.5 and σ = 0.25, 1, 4. Your answer
# should be a single plot with three panels (each one for a different
# value of σ). 

# Plot the functions
p1 = plot_production([10, 10], 0.5, 0.25) # s = 0.25
p2 = plot_production([10, 10], 0.5, 1) # s = 1
p3 = plot_production([10, 10], 0.5, 4) # s = 4
plot(p1, p2, p3, layout = (3,1))

# 4.3  Write a function that takes α, σ, w1, w2, and y as arguments 
# and returns the cost function and x1 and x2. 
# Inside this function, you will have to solve a 
# constrained minimization problem (because you have 
# nonnegativity constraints). 


# Define the CES production function
function optimize_cost(a,s,w1, w2, y)
    # once again define the function
    function production(x, a, s) 
        if s == 1  
            return((x[1].^a) .* (x[2].^(1-a))).^0.5
        else
            return (a .* x[1].^((s-1)/s) + (1-a) .* x[2].^((s-1)/s)).^(s/(s-1))
        end
    end
    
    function obj_fun(x, w1, w2, y, a, s, lambda)
        # Cost function
        cost = w1 * x[1] + w2 * x[2]
        # penalty
        pen = (production(x, a, s) - y)^2
        # Penalized objective
        return cost + lambda * pen
    end
    # Main optimization function
    function optimize_ces(w1, w2, y, a, s)
        # Starting point for optimization
        initial_x = [1.0, 1.0]  # Initial guesses for x1 and x2
        
        # Define bounds for x1 and x2 
        lower_bounds = [1e-6, 1e-6] # zero does not work and makes it impossible to run!
        upper_bounds = [Inf, Inf]
        
        # Penalty factor for constraint violation
        lambda = 1000.0 # parameter 
        
        # Create the objective function closure
        objective = x -> obj_fun(x, w1, w2, y, a, s, lambda)
        
        # Use Fminbox for optimisation
        result = optimize(
            objective,
            lower_bounds,
            upper_bounds,
            initial_x,
            Fminbox(BFGS())
        )
        # Extract optimal values
        optimal_x = Optim.minimizer(result)
        optimal_cost = Optim.minimum(result)
        return optimal_x, optimal_cost
end
        
        # Optimise 
        optimize_ces(w1, w2, y, a, s)
        optimal_x, optimal_cost = optimize_ces(w1, w2, y, a, s)
        return optimal_x, optimal_cost
    end
end


# test 4.3
optimize_cost(0.5,0.5, 1, 1, 10)

# 4.4. Plot the cost function and the input demand functions (x1 and x2)
# for three different values of σ: σ = 0.25, 1, 4 as a function of w1 for
# w2 = 1 and y = 1. Set α to 0.5. Your answer should be a single plot
# with three panels (cost, x1, and x2) and three lines in each panel
# (each one for a different value of σ). You do not need to write a
# function for this part, it is enough to write a script that first calls
# the functions you wrote in the first part and then plots the results

optimize_cost(0.5, 0.25, 1, 1, 1)
optimize_cost(0.5, 1, 1, 1, 1)
optimize_cost(0.5, 4, 1, 1, 1)

# To plot, we need to calculate the values of the function for different parameters 
# Define the grid for w1
grid_w1 = 0.1:0.5:10 # We start with w1 ~0, because we do not want price equal to zero (precision)

# To store results
results25 = DataFrame(w1=Float64[], x1=Float64[], x2=Float64[], cost=Float64[])
results1 = DataFrame(w1=Float64[], x1=Float64[], x2=Float64[], cost=Float64[])
results4 = DataFrame(w1=Float64[], x1=Float64[], x2=Float64[], cost=Float64[])


# Loop results for sigma = 0.25
for w1 in grid_w1
    optimal_x, optimal_cost = optimize_cost(0.5, 0.25, w1, 1, 1)
    push!(results25, (w1, optimal_x[1], optimal_x[2], optimal_cost))
end

# Loop results for sigma = 1
for w1 in grid_w1
    optimal_x, optimal_cost = optimize_cost(0.5, 1, w1, 1, 1)
    push!(results1, (w1, optimal_x[1], optimal_x[2], optimal_cost))
end

# Loop results for sigma = 4
for w1 in grid_w1
    optimal_x, optimal_cost = optimize_cost(0.5, 4, w1, 1, 1)
    push!(results4, (w1, optimal_x[1], optimal_x[2], optimal_cost))
end

# Plot results for s=0.25
p25_1 = plot(results25.x1, results25.w1, legend=false, levels = 50, title = "x1 vs w1")
p25_2 = plot(results25.x2, results25.w1, legend=false, levels = 50, title = "x2 vs w1")
p25_3 = plot(results25.cost, results25.w1, legend=false, levels = 50, title = "cost vs w1")
plot(p25_1, p25_2, p25_3, layout = (3,1))

# Plot results for s = 1
p1_1 = plot(results1.x1, results1.w1, legend=false, levels = 50, title = "x1 vs w1")
p1_2 = plot(results1.x2, results1.w1, legend=false, levels = 50, title = "x2 vs w1")
p1_3 = plot(results1.cost, results1.w1, legend=false, levels = 50, title = "cost vs w1")
plot(p1_1, p1_2, p1_3, layout = (3,1))

# Plot results for s = 4
p4_1 = plot(results4.x1, results4.w1, legend=false, levels = 50, title = "x1 vs w1")
p4_2 = plot(results4.x2, results4.w1, legend=false, levels = 50, title = "x2 vs w1")
p4_3 = plot(results4.cost, results4.w1, legend=false, levels = 50, title = "cost vs w1")
plot(p4_1, p4_2, p4_3, layout = (3,1))