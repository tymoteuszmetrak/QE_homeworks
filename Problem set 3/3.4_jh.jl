using LinearAlgebra, Distributions, Plots, Random, QuantEcon

# Transition matrix P for Z_t
P = [0.5 0.3 0.2;
     0.2 0.7 0.1;
     0.3 0.3 0.4]

# Define state spaces
Z_states = [1, 2, 3]  # Corresponding to z1, z2, z3
X_states = 0:5

# Define the policy function σ(X_t, Z_t)
function sigma(X, Z)
    if Z == 1
        return 0
    elseif Z == 2
        return X
    elseif Z == 3 && X <= 4
        return X + 1
    else
        return 3
    end
end

# Create the joint transition matrix for {X_t, Z_t}
function joint_transition_matrix(P, X_states, Z_states)
    N_X = length(X_states)
    N_Z = length(Z_states)
    joint_matrix = zeros(N_X * N_Z, N_X * N_Z)
    
    for (i, X) in enumerate(X_states) #index i = 1: X = 0, index i = 2: X = 1,...
        for (j, Z) in enumerate(Z_states) #index j = 1: Z = 1, index j = 2: Z = 2,...
            for (k, X_next) in enumerate(X_states) #index k = 1: X_next = 0, index k = 2: X_next = 1,....
                for (l, Z_next) in enumerate(Z_states) #index l = 1: Z = 1, index j = 1: Z = 2,...
                    if sigma(X, Z) == X_next
                        joint_matrix[(i - 1) * N_Z + j, (k - 1) * N_Z + l] = P[j, l] #pairs (X,Z) in first row (for current (X,Z) = (0,1)) are in order: (0,1), (0,2),(0,3),(1,1),(1,2),(1,3),(2,1),...
                        #j,l, not i,k because probabilities change with Z, not X 
                    end
                end
            end
        end
    end
    return joint_matrix
end

# Obtain the joint transition matrix
P_joint = joint_transition_matrix(P, X_states, Z_states)

P_stationary = P_joint^1000 #approximate stationary P for a very large t


#####################################################################################

ρ = 0.9 #with lower - process shrinks faster -> smaller grid needed
σ = 0.02
N_states = 18
tauch_approximation_1 = tauchen(N_states,ρ, σ)
tauch_approximation_1.p
tauch_approximation_1.state_values #range syntax

state_space_1 = collect(tauch_approximation_1.state_values) 



function stationary_distribution(P_joint)
    vals, vecs = eigen(P_joint')
    stationary = vecs[:, argmax(abs.(vals .- 1.0))]  # Find eigenvector for eigenvalue 1
    stationary = stationary ./ sum(stationary)  # Normalize to sum to 1
    return stationary
end

a = stationary_distribution(stationary_dist)


A = [2 2; 2 3]
B = A && [A == 3]

realeigen(P) = eigen(P) && !(typeof(eigen(P).vectors[i]) == Complex64 && !(typeof(eigen(P).values[i]) == Complex64


function joint_stat_dist(P)
    for i in eigen(P).vectors  && !(typeof(eigen(P).vectors[i]) == Complex64
        if !(typeof(eigen(P).values[i]) == Complex64) && eigen(P).values[i] - 1 < 1e-8
        return eigen(P).vectors[i]
        end
    end
end

joint_stat_dist(stationary_dist)


eigen(stationary_dist).vectors[eigen(stationary_dist) == 1]

function joint_stat_dist(stationary_dist)


print(data.values)
print(data.vectors)

############################################

# Calculate the stationary distribution

# Marginalize to get stationary distribution of X_t
function marginal_distribution(stationary_dist, N_X, N_Z)
    marginal = zeros(N_X)
    for i in 1:N_X
        marginal[i] = sum(stationary_dist[(i - 1) * N_Z + 1:i * N_Z])
    end
    return marginal
end

marginal_X = marginal_distribution(stationary_dist, length(X_states), length(Z_states))

# Expected value of X_t
expected_X = sum(marginal_X .* X_states)

# Output results
println("Stationary distribution of X_t: ", marginal_X)
println("Expected value of X_t: ", expected_X)