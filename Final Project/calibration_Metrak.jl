# Packages
using Distributions, QuantEcon
using Roots

#####################################################
# Step 1
######################################################
# Discretize the Productivity Process
######################################################

# Parameters for the productivity process
ρ = 0.9       # Persistence parameter
σ = 0.4       # Standard deviation of the shock
N_z = 5       # Number of grid points for productivity
μ_z = 0.0     # Mean of the log productivity process (to be normalized)

# Discretize the AR(1) process using Tauchen's method
mc_z = tauchen(N_z, ρ, σ, μ_z)

# Extract the discretized values and transition matrix
z_grid = exp.(mc_z.state_values)  # Convert from log to levels
P_z = mc_z.p                      # Transition matrix

# Normalize the productivity grid so that the average productivity is 1
z_grid_normalized = z_grid / sum(z_grid .* stationary_distributions(mc_z)[1])

# Display the results
println("Discretized productivity grid (normalized): ", z_grid_normalized)
println("Transition matrix: ")
display(P_z)



#####################################################
# Step 2
######################################################
# Set the value of α using the labor share
######################################################

labor_share_US = 0.60  # Labor share in the U.S. economy (from Penn World Table)
α = 1 - labor_share_US  # Output elasticity of capital

# Display the result
println("Labor share in the U.S. economy: ", labor_share_US)
println("Output elasticity of capital (α): ", α)


#####################################################
# Step 3
######################################################
#  Calibrate Remaining Parameters
######################################################

# Known parameters
α = 0.40       # Output elasticity of capital (from Step 2)
r = 0.04       # Interest rate
w = 1.0        # Wage rate
L = 1.0        # Labor supply (normalized)
investment_to_output_ratio = 0.2
govt_to_output_ratio = 0.2


# Step 3.1: Solve for G and τ
# I can do it analytically on the piece of paper
# 1-alpha = 0.6; output 1/0.6 = 5/3; G = 0.2*output -> G = 1/3;  
# for λ = 0; G = τ*y_dashed, but since y_dashed is 1 and G is 1/3, that implies tau = 1/3
G = 1/3
τ = 1/3
    
println("Goverment expenditure (G): ", G)
println("Tax rate (τ): ", τ)

# Step 3.2: Solve for A, δ, and K
# Again, I can do it analytically on the piece of paper
# After some calculations I receive:
δ = 0.04
A = 0.7137
K = 25/3

println("Capital stock (K): ", K)
println("Depreciation rate (δ): ", δ)
println("Productivity (A): ", A)


# Step 3.3: Solve for β (requires solving the household problem)
# This step is computationally intensive and involves solving the Bellman equation.
# You will need to iterate over β until the asset market clears (aggregate asset demand = K).

using LinearAlgebra, Interpolations

# Calibrated parameters
α = 0.40       # Output elasticity of capital
r = 0.04       # Interest rate
w = 1.0        # Wage rate
L = 1.0        # Labor supply (normalized)
K = 25/3       # Capital stock
δ = 0.04       # Depreciation rate
A = 0.7137     # Productivity
τ = 1/3        # Tax rate

# Household parameters
γ = 2.0        # Risk aversion
ϕ = 0.0        # Borrowing constraint
N_a = 100      # Grid size for assets
a_min = -ϕ     # Minimum assets
a_max = 2 * K  # Maximum assets

# Asset grid
a_grid = range(a_min, a_max, length=N_a)

# Initial guess for β
β_guess = 0.92

# Utility function
# CRRA (Constant Relative Risk Aversion) utility
function utility(c)
    return c > 0 ? (c^(1 - γ) - 1) / (1 - γ) : -1e10 # If consumption c is non-positive, assigns a very large negative utility (-1e10)
end

# Budget constraint
function budget_constraint(a, y, r, w)
    return (1 + r) * a + (1 - τ) * y * w
end

# Value function iteration
function solve_household_problem(β, tol=1e-6, max_iter=1000)
    # Uses Bellman’s Equation to solve for the optimal savings policy.
    V_old = zeros(N_a)
    V_new = similar(V_old)
    policy = zeros(Int, N_a)
    
    for iter in 1:max_iter
        #Loops over all asset values in the grid.
        for (i, a) in enumerate(a_grid)
            max_val = -Inf
            best_choice = 1
            
            for (j, a_next) in enumerate(a_grid)
                c = budget_constraint(a, L, r, w) - a_next
                val = utility(c) + β * V_old[j]
                
                if val > max_val
                    max_val = val
                    best_choice = j
                end
            end
            
            V_new[i] = max_val
            policy[i] = best_choice
        end
        
        if maximum(abs.(V_new - V_old)) < tol
            # If the difference between old and new value functions is small enough (tol), stop iterating.
            break
        end
        
        V_old .= V_new
    end
    
    return V_new, policy
end

# Find equilibrium β
function find_beta(target_K, β_low=0.80, β_high=0.99, tol=1e-4)
    # Uses a bisection method to find the value of β that ensures aggregate savings match the capital stock K
    while β_high - β_low > tol
        β_mid = (β_low + β_high) / 2
        _, policy = solve_household_problem(β_mid)
        
        # Compute aggregate assets
        agg_assets = mean(a_grid[policy])
        
        if agg_assets > target_K
        # If savings exceed K, β is too high (households are too patient) -> decrease β
            β_high = β_mid
        else
        # Otherwise, increase β
            β_low = β_mid
        end
    end
    return (β_low + β_high) / 2
end

# Solve for β
β_calibrated = find_beta(K)
println("Calibrated β: ", β_calibrated)



############################################
# Comparing economies 
############################################


# Known parameters when λ=0.15

# β around 0.9664  # Discount factor
α = 0.40       # Output elasticity of capital
δ = 0.04       # Depreciation rate
A = 0.7137     # Productivity
γ = 2.0        # Risk aversion
ϕ = 0.0        # Borrowing constraint
ρ = 0.9        # Persistence parameter in productivity
σ = 0.4        # Standard deviation of the shock

# P_z or z_grid_normalized

