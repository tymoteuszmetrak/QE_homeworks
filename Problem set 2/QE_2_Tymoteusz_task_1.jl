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
using Plots
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