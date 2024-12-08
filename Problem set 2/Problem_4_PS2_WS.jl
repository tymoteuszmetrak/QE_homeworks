# Problem 4 - Production

using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, NLsolve, Optim, Roots, Calculus

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
