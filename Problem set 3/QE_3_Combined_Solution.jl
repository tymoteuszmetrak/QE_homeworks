# PS3
# Jan Hincz, Tymoteusz Mętrak, Wojciech Szymczak

# Read libraries 
using Plots, Statistics, LinearAlgebra, Parameters, PrettyTables, LinearAlgebra, Distributions, Random, QuantEcon


#PROBLEM 1


# Write a function that solves the Bellman equation. There is more
# than one way to do this. You can either exploit the fact that the
# problem is finite and use backward induction, or you can use the
# value function iteration algorithm. To be more precise: it is possible
# o use the value function iteration algorithm that converges
# in just one iteration if you start with v(N) and then consider
# v(N − 1), v(N − 2), . . . , v(0).

# For description of the Bellman Equations please find solution in pdf. 
function bell_eq(N, X, p_min, p_max, C, q, f)
    # Generate possible prices
    p_change = 0.1 # as in exercise 
    prices = collect(p_min:p_change:p_max)  

    # Lists for storing 
    v = zeros(N + 1) # Value function
    sig1 = zeros(Bool, N + 1) #Approach?
    sig2 = zeros(Bool, N + 1, length(prices)) # Buy, given price?
    prob = zeros(N + 1) # Probability of choosing a price
    exp_price = zeros(N + 1) # Expected price for each n

    # Compute over number of vendors a la backward induction (finite problem)
    for n in N:-1:1
        # Compute value of termination 
        v_T = -(n - 1) * f # here is n-1 because I make decision at this stage, after approaching n-1 (see PDF)

        # Compute expected value of buying
        E_vB = 0.0
        for (i, p) in enumerate(prices)
            v_B = X - p - C - n * f # Value of buying at price p; did not know how to write the costs different than C (administration), f(mental cost)
            sig2[n, i] = v_B > v[n] # buy if v_B > value
            E_vB += max(v_B, v[n]) * (1 / length(prices))  # Equal price probabilities;from uniform distribution the probability is equal; as enumerating over prices, the length used 
        end

        # Compute choices and values
        v_A = -f + q * E_vB + (1 - q) * v[n] # Value of approaching the next vendor
        sig1[n] = v_A > v_T # Policy: approach vendor if v_A > v_T
        v[n] = max(v_A, v_T)  # Update value function

        # Compute probability and expected price
        prob[n] = sum(sig2[n, :]) / length(prices) 
        if prob[n] > 0
            exp_price[n] = sum(prices[i] * sig2[n, i] for i in 1:length(prices)) / sum(sig2[n, :])
        end
    end

    return v, sig1, sig2, prob, exp_price
end

# Parameter from the exercise (point3)
N = 50 # Number of vendors 
X = 50 # Maximum willingness to pay
p_min = 10.0 # Min price
p_max = 100.0 # Max price
C = 0.5 # Cost (assumed transaction cost)
q = 0.15 # Probability a vendor has the orchid
f = 0.05 # Mental cost (i did not know; assumed 10% of administrative cost)

# Solve Bellman equation
val, pol1, pol2, probab, price = bell_eq(N, X, p_min, p_max, C, q, f)

# What is the probability that Basil will buy the orchid?
mean(probab)

# what is the expected price that Basil will pay for the orchid (conditional on actually buying it)?
dot(price, probab)/sum(probab) # weighted mean 
# Multiply each probability by the expected price 

# what is the expected number of vendors Basil will approach?
dot(0:1:50,pol1)/sum(pol1) 
# Multiply the number (order) of each vendor by the decision (approach, terminate) 

#(d) is Basil more or less willing to agree for a higher price as he
#approaches more vendors?
plot(0:1:50, price) # As Basill approaches more vendors he is willing only to accept lower prices



#PROBLEM 2

# Here I inititate the function for the job search (with parametrs the same as defined in class)
function create_job_search_model(;
    n = 100, # wage grid size - number of possible wage offers
    w_min = 10.0, # lowest wage
    w_max = 60.0, # highest wage
    a = 200, # wage distribution parameter
    b = 100, # wage distribution parameter
    β = 0.96, # discount factor
    c = 10.0,
    p = 0.5 # unemployment compensation
    )
    w_vals = collect(LinRange(w_min, w_max, n)) #n
    ϕ = pdf(BetaBinomial(n-1, a, b)) #n-1 vs. n
    return (; n, w_vals, ϕ, β, c,p)
end

my_model = create_job_search_model()


### APPROACH I: directly solve for h*
get_h_new(h_old; n, w_vals, ϕ, β, c, p) = 
    c + β * sum(ϕ[i] * max(h_old, (1 - p) * w_vals[i] / (1 - β) + p * c / (1 - β)) for i in 1:n)

# As compared to the code from classes, now the value is expressed as:
# with probablity (1-p) gets wage (not seperated from job) and with probability p gets unemployment compensation 

wrap_h_new(h_old) = get_h_new(h_old; my_model..., c = c, p = p) # added parameter for p
#iteratively solving to find h -> solution where h_old = h_new ()

# Function for h1
function get_h_1(model; tol=1e-8, maxiter = 1000, h_init = model.c / (1 - model.β))
    (; n, w_vals, ϕ, β, c,p) = model  # Unpack model parameters

    h_old = Float64(h_init)
    h_new = h_old
    h_history = [h_old]
    error = tol + 1.0
    iter = 1

    while error > tol && iter < maxiter
        h_new = get_h_new(h_old; n = n, w_vals = w_vals, ϕ = ϕ, β = β, c = c, p = p)
        error = abs(h_new - h_old)
        h_old = h_new
        push!(h_history, h_old)
        iter += 1
    end

    return h_new, iter, error, h_history
end

h, iter, error, h_history = get_h_1(my_model)

# Iterate over p to produce the relationship between reservation wage and job seperation probability
p_vec = LinRange(0, 1.1, 100)
# I assumed for purpose that the probability of job separation can exceed 1
# Why? Because in these circumstances I can check that one nevers decides to work (because unemployment compensation is higher than the value)

 
# 1. Create a plot that shows how the reservation wage w* changes with p




function get_v_from_h(model,h) #slides 45-46
    (; n, w_vals, ϕ, β, c,p) = model # unpack the model parameters
    σ = (1 .- p) .* w_vals ./ (1 - β) .+ p * c ./ (1 - β) .>= h # this is a vector of booleans; σ = 1 when it's optimal to accept, σ = 0 when it's optimal to reject
    v = σ .* ((1 .- p) .* w_vals ./ (1 - β) .+ p * c ./ (1 - β)) + (1 .- σ) .* h # this is a vector of floats - see slide 46
    return v, σ
end

v, σ = get_v_from_h(my_model,h)

reservation_wage_vec = []
# Iterate 
for p in p_vec
    my_model = create_job_search_model(;p=p)
    h, iter, error, h_history = get_h_1(my_model)
    v, σ = get_v_from_h(my_model, h)
    push!(reservation_wage_vec, my_model.w_vals[σ][1])
end

# Plot the result 
plot(p_vec, reservation_wage_vec, xlabel="Job separation probability (p)", ylabel="Reservation wage (w*)", linewidth=4)


# 2. Calculate the probability that an unemployed worker will accept a job. 
# This probability is equal to the probability that w ≥ w∗
# Call it q. Create a plot that shows how q changes with p. 

p_vec = LinRange(0, 0.9, 1000)  # Range of job separation probabilities


function get_q(model, h)
    (; n, w_vals, ϕ, β, c,p) = model  # Unpack model parameters
    # Calculate the threshold vector for reservation wage
    threshold = w_vals ./ (1 - β)   # Element-wise addition
    σ = threshold .>= h  # Boolean mask for w >= w*

    # Sum probabilities for wages satisfying the reservation wage condition
    acceptance_prob = sum(ϕ[findall(σ)])  # `findall(σ)` gets the indices where σ is true
    return acceptance_prob
end

# Iterate over probability of job separation
q_vec = []  # Store probabilities of accepting a job
for p in p_vec
    my_model = create_job_search_model(;p=p)
    h, iter, error, h_history = get_h_1(my_model)  # Solve for reservation wage
    q = get_q(my_model, h)  # probability that workers accepts job
    push!(q_vec, q) # save to vector 
end

# Plot q vs p
plot(p_vec, q_vec, xlabel="Job separation probability (p)", ylabel="Acceptance probability (q)", linewidth=4, label=false)


# 3. Note that an unemployed worker will stay unemployed for exactly
#one period with the probability (1 − q) q, for two periods with the
# probability (1 − q)2 q, and so on. 

# This is a geometric probability.
# Calculate the expected duration of unemployment. Create a plot
# that shows how the expected duration of unemployment changes with p

# Note that the expected duration would be presented analytically as:
# EN = 1 * (1-q)*q + 2 * (1-q)2*q + 3 * (1-q)3*q + ... + n*(1-q)^n * q 
# EN = 1/q 

# Initialize a vector to store expected durations
duration_vec = []  # Store expected durations
p_vec = LinRange(0, 0.5, 50)  # Range of p values

# Loop over p values
for p in p_vec
    # Solve for reservation wage h given p
    my_model = create_job_search_model(;p=p)
    h, iter, error, h_history = get_h_1(my_model)
    
    # Compute the probability of accepting a job
    q = get_q(my_model, h)
    
    # Calculate expected duration of unemployment
    en = 1 / q
    push!(duration_vec, en)
end

# Plot expected duration vs p
plot(p_vec, duration_vec, xlabel="Job separation probability (p)", ylabel="Expected duration of unemployment", linewidth=4, label=false)



#PROBLEM 3: Convergence in the Neoclassical Growth Model


@with_kw struct NGMProblem2 #@with_kw: macro of Parameters package

    β = 0.95 # discount factor
    α = 0.3 # production function parameter
    δ = 0.05 # depreciation rate
    γ = 2.0 # intertemporal elasticity of substitution (inverse)

    f = x -> x^α # production function
    u = γ == 1 ? c -> log(c) : c -> c ^ (1-γ) / (1-γ)   # CRRA utility function
    k_star = ((β^(-1) - 1 + δ) / α) ^(1/(α-1)) # steady state capital

    k_min = 0.5 * k_star # minimum capital
    k_max = 1.25*k_star # maximum capital
    
    n = 100 # number of grid points
    k_grid = range(k_min,stop=k_max,length=n) #n-element grid for capital from k_min to k_max
    
end


### NGM PROBLEM2

function T(v,model) # Bellman operator
    @unpack n, k_grid, β, α, δ, f, u = model

    #initialise the function
    v_new = zeros(n)
    reward = zeros(n,n) 
    σ = zeros(n)

        for (k_index,k) in enumerate(k_grid) # loop over capital today
            for (k_next_index, k_next) in enumerate(k_grid) # loop over capital tomorrow

                c = k^α - k_next + (1-δ)*k # consumption
                if c > 0
                    reward[k_index,k_next_index] = u(c) + β * v[k_next_index] #v(k) entry
                else
                    reward[k_index,k_next_index] = -Inf #penalty instead of constrained optimisation with c>0
                end

            end 

            v_new[k_index], k_next_index_opt = findmax(reward[k_index,:]) 
            σ[k_index] = k_grid[k_next_index_opt]
        end
        
    return v_new, σ
end


function vfi(model;maxiter=1000,tol=1e-8) # value function iteration
    @unpack n, k_grid, β, α, δ, f, u = model
    v_init = zeros(n); err = tol + 1.0; iter = 1 #initial guess
    v = v_init
    v_history = [v_init]
    σ = zeros(n)
    while err > tol && iter < maxiter
        v_new, σ = T(v,model)
        err = maximum(abs.(v_new - v)) 
        push!(v_history,v_new)
        v = v_new
        iter += 1
    end


    return v, σ, iter, err, v_history
end


my_ngm_low_γ = NGMProblem2(n=300,γ=0.5)
v_low_γ, σ_low_γ, iter_low_γ, err_low_γ, v_history_low_γ = vfi(my_ngm_low_γ)

my_ngm_mid_γ = NGMProblem2(n=300,γ=1)
v_mid_γ, σ_mid_γ, iter_mid_γ, err_mid_γ, v_history_mid_γ = vfi(my_ngm_mid_γ)

my_ngm_high_γ = NGMProblem2(n=300,γ=2)
v_high_γ, σ_high_γ, iter_high_γ, err_high_γ, v_history_high_γ = vfi(my_ngm_high_γ)

#PLOTS - to check whether they meet economic intuition

plot_v_low_γ = plot(my_ngm_low_γ.k_grid,v_low_γ, label="v(k)",linewidth=4,xlabel = "k",ylabel = "v");
plot_σ_low_γ = plot(my_ngm_low_γ.k_grid,σ_low_γ, label="policy: k'(k)", linewidth=4,xlabel = "k",);

# add the 45 degree line
plot!(my_ngm_low_γ.k_grid,my_ngm_low_γ.k_grid, label="45 degree line",linewidth=2,linestyle=:dash);

# add the steady state
vline!([my_ngm_low_γ.k_star], label="steady state",linewidth=2,linestyle=:dash);
plot(plot_v_low_γ,plot_σ_low_γ,layout=(1,2),legend=:topleft)


plot_v_mid_γ = plot(my_ngm_mid_γ.k_grid,v_mid_γ, label="v(k)",linewidth=4,xlabel = "k",ylabel = "v");
plot_σ_mid_γ = plot(my_ngm_mid_γ.k_grid,σ_mid_γ, label="policy: k'(k)", linewidth=4,xlabel = "k",);

# add the 45 degree line
plot!(my_ngm_mid_γ.k_grid,my_ngm_mid_γ.k_grid, label="45 degree line",linewidth=2,linestyle=:dash);

# add the steady state
vline!([my_ngm_mid_γ.k_star], label="steady state",linewidth=2,linestyle=:dash);
plot(plot_v_mid_γ,plot_σ_mid_γ,layout=(1,2),legend=:topleft)


plot_v_high_γ = plot(my_ngm_high_γ.k_grid,v_high_γ, label="v(k)",linewidth=4,xlabel = "k",ylabel = "v");
plot_σ_high_γ = plot(my_ngm_high_γ.k_grid,σ_high_γ, label="policy: k'(k)", linewidth=4,xlabel = "k",);

# add the 45 degree line
plot!(my_ngm_high_γ.k_grid,my_ngm_high_γ.k_grid, label="45 degree line",linewidth=2,linestyle=:dash);

# add the steady state
vline!([my_ngm_high_γ.k_star], label="steady state",linewidth=2,linestyle=:dash);
plot(plot_v_high_γ,plot_σ_high_γ,layout=(1,2),legend=:topleft)

#PLOT FOR POLICY FOR ALL GAMMAS
plot(my_ngm_low_γ.k_grid,σ_low_γ, label="γ = 0.5", linewidth=4,xlabel = "k", ylabel = "k'")
plot!(my_ngm_mid_γ.k_grid,σ_mid_γ, label="γ = 1", linewidth=4,xlabel = "k", ylabel = "k'")
plot!(my_ngm_high_γ.k_grid,σ_high_γ, label="γ = 2", linewidth=4,xlabel = "k", ylabel = "k'")
vline!([my_ngm_high_γ.k_star], label="steady state",linewidth=2,linestyle=:dash)



# obtain a sample path for the capital stock

Time = 300
k_path_low_γ = zeros(Time)
k_path_mid_γ = zeros(Time)
k_path_high_γ = zeros(Time)
k_path_low_γ[1] = my_ngm_low_γ.k_grid[1] # start at the lowest level of capital
k_path_mid_γ[1] = my_ngm_mid_γ.k_grid[1] # start at the lowest level of capital
k_path_high_γ[1] = my_ngm_high_γ.k_grid[1] # start at the lowest level of capital

for i in 2:Time
    k_path_low_γ[i] = σ_low_γ[findfirst(x->x==k_path_low_γ[i-1],my_ngm_low_γ.k_grid)]
    k_path_mid_γ[i] = σ_mid_γ[findfirst(x->x==k_path_mid_γ[i-1],my_ngm_mid_γ.k_grid)]
    k_path_high_γ[i] = σ_high_γ[findfirst(x->x==k_path_high_γ[i-1],my_ngm_high_γ.k_grid)]
end


#Plots of convergence
plot(1:Time,k_path_low_γ, label="k(t); γ = 0.5 ",linewidth=4,xlabel = "t",ylabel = "k")
plot!(1:Time,k_path_mid_γ, label="k(t); γ = 1",linewidth=4,xlabel = "t",ylabel = "k")
plot!(1:Time,k_path_high_γ, label="k(t); γ = 2",linewidth=4,xlabel = "t",ylabel = "k")


period_low_γ = findfirst(x -> my_ngm_low_γ.k_star - x < 0.5 * (my_ngm_low_γ.k_star - my_ngm_low_γ.k_min), k_path_low_γ) #6
period_mid_γ = findfirst(x -> my_ngm_mid_γ.k_star - x < 0.5 * (my_ngm_mid_γ.k_star - my_ngm_mid_γ.k_min), k_path_mid_γ) #8
period_high_γ = findfirst(x -> my_ngm_high_γ.k_star - x < 0.5 * (my_ngm_high_γ.k_star - my_ngm_high_γ.k_min), k_path_high_γ) #11


periods = [period_low_γ,period_mid_γ,period_high_γ]
γ_vec = [0.5,1.0,2.0]

data = hcat(γ_vec,periods)
    
header = (["γ","time of half-convergence"],["","periods"]) #[columns], [subtitles]

pretty_table(data;
header=header,
header_crayon=crayon"yellow bold" ,
formatters = ft_printf("%5.2f",0),
display_size =  (-1,-1)) #to get the entire table


#3.1. FUNCTION WITH TABLE WITH HALF CONVERGENCE

function half_convergence(n::Int64,γ_vec::Vector)
    periods = Int[]
    for γ in γ_vec
        model = NGMProblem2(γ=γ)
        _, σ, _, _, _ = vfi(model)
        k_path = zeros(n)
        k_path[1] = model.k_grid[1]
        for i in 2:n
            k_path[i] = σ[findfirst(x -> x == k_path[i-1], model.k_grid)]
        end
        period = findfirst(x -> model.k_star - x < 0.5 * (model.k_star - model.k_grid[1]), k_path)
        push!(periods, period)
    end
    
    data = hcat(γ_vec,periods)
    
    header = (["γ","time of half-convergence"],["","periods"]) #[columns], [subtitles]

    table = pretty_table(data;
    header=header,
    header_crayon=crayon"yellow bold" ,
    formatters = ft_printf("%5.2f",0),
    display_size =  (-1,-1)) #to get the entire table
    return table
end

half_convergence(300,[0.5,1.0,2.0]) #6,8, 12 - seems like some rounding issue for γ = 2, because for γ = 1.999 it's still 11 like above (line 153) 
half_convergence(300,[0.5,1.0,1.999]) #6,8,11

#Plots
plot_capital = plot(1:Time,k_path_low_γ, label="k(t); γ = 0.5 ",linewidth=4,xlabel = "t",ylabel = "k")
plot!(1:Time,k_path_mid_γ, label="k(t); γ = 1",linewidth=4,xlabel = "t",ylabel = "k")
plot!(1:Time,k_path_high_γ, label="k(t); γ = 2",linewidth=4,xlabel = "t",ylabel = "k")


plot_output = plot(1:Time,k_path_low_γ.^my_ngm_low_γ.α, label="y(t); γ = 0.5 ",linewidth=4,xlabel = "t",ylabel = "y")
plot!(1:Time,k_path_mid_γ.^my_ngm_mid_γ.α, label="y(t); γ = 1 ",linewidth=4,xlabel = "t",ylabel = "y")
plot!(1:Time,k_path_high_γ.^my_ngm_high_γ.α, label="y(t); γ = 2 ",linewidth=4,xlabel = "t",ylabel = "y")


function i_ratio(k_path::Vector,model)
    i_ratio = zeros(length(k_path) - 1)
    for i in 1:(length(k_path) - 1)
    i_ratio[i] = (k_path[i+1] - (1-model.δ)*k_path[i])/(k_path[i]^model.α)
    end
    return i_ratio
end

i_ratio_low_γ =  i_ratio(k_path_low_γ,my_ngm_low_γ)
i_ratio_mid_γ =  i_ratio(k_path_mid_γ,my_ngm_mid_γ)
i_ratio_high_γ =  i_ratio(k_path_high_γ,my_ngm_high_γ)


plot_i_ratio = plot(1:(Time-1),i_ratio_low_γ, label="i_ratio(t); γ = 0.5 ",linewidth=4,xlabel = "t",ylabel = "i ratio")
plot!(1:(Time-1),i_ratio_mid_γ, label="i_ratio(t); γ = 1 ",linewidth=4,xlabel = "t",ylabel = "i ratio")
plot!(1:(Time-1),i_ratio_high_γ, label="i_ratio(t); γ = 2 ",linewidth=4,xlabel = "t",ylabel = "i ratio")

plot_c_ratio = plot(1:(Time-1),1 .- i_ratio_low_γ, label="c_ratio(t); γ = 0.5 ",linewidth=4,xlabel = "t",ylabel = "c ratio")
plot!(1:(Time-1),1 .-i_ratio_mid_γ, label="c_ratio(t); γ = 1 ",linewidth=4,xlabel = "t",ylabel = "c ratio")
plot!(1:(Time-1),1 .-i_ratio_high_γ, label="c_ratio(t); γ = 2 ",linewidth=4,xlabel = "t",ylabel = "c ratio")

plot(plot_capital,plot_output,plot_i_ratio,plot_c_ratio,layout=(2,2),legend=:topright)
savefig("convergence_plots_1.pdf")

#3.2. FUNCTION WITH CONVERGENCE PLOTS

function convergence_plots(Time::Int64,γ_vec::Vector)
    plot_capital = plot()
    plot_output = plot()
    plot_i_ratio = plot()
    plot_c_ratio = plot()
    for γ in γ_vec
        model = NGMProblem2(γ=γ)
        _, σ, _, _, _ = vfi(model)
        k_path = zeros(Time)
        k_path[1] = model.k_grid[1]

        for i in 2:Time
            k_path[i] = σ[findfirst(x -> x == k_path[i-1], model.k_grid)]
        end
        i_ratio = zeros(length(k_path) - 1)

        for i in 1:(length(k_path) - 1)
        i_ratio[i] = (k_path[i+1] - (1-model.δ)*k_path[i])/(k_path[i]^model.α)
        end
        
        plot_capital = plot!(plot_capital,1:Time,k_path, label="k(t); γ = $γ",linewidth=4,xlabel = "t",ylabel = "k")
        plot_output = plot!(plot_output,1:Time,k_path.^model.α, label="y(t); γ = $γ",linewidth=4,xlabel = "t",ylabel = "y")
        plot_i_ratio = plot!(plot_i_ratio,1:(Time-1),i_ratio, label="i_ratio(t); γ = $γ",linewidth=4,xlabel = "t",ylabel = "i ratio")
        plot_c_ratio = plot!(plot_c_ratio,1:(Time-1),1 .- i_ratio, label="c_ratio(t); γ = $γ",linewidth=4,xlabel = "t",ylabel = "c ratio")
    end
    
    return plot(plot_capital,plot_output,plot_i_ratio,plot_c_ratio, layout=(2,2),legend=:topright)
end

convergence_plots(300,[0.5,1.0,2.0])
savefig("convergence_plots_2.pdf")
#figures preserve the ranking, but once again function seems to skew the results a bit vs. outside of function

#quicker convergence for lower γ - less concave function - lower consumption smoothing motive
#-> will get quicker to steady state of c and k by in the meantime saving more and sacrificing c


###PROBLEM 4


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
                        joint_matrix[(i - 1) * N_Z + j, (k - 1) * N_Z + l] = P[j, l] #pairs (X,Z) in first row (for current (X,Z) = (0,1)) are in order for next period: (0,1), (0,2),(0,3),(1,1),(1,2),(1,3),(2,1),...
                        #j,l, not i,k because probabilities change with Z, not X 
                    end
                end
            end
        end
    end
    return joint_matrix
end

#4.1 Obtain the joint transition matrix
P_joint = joint_transition_matrix(P, X_states, Z_states)

P_stationary = P_joint^1000 #approximate stationary P for a very large t

#4.2a
joint_stationary_distr = P_stationary[1, :] ./ sum(P_stationary[1, :])  # Normalize from any row (potential numerical imprecision)
#we get joint stationary distribution for pairs (0,1), (0,2),(0,3),(1,1),(1,2),(1,3),(2,1),...

#4.2b
marginal_X0 = sum(joint_stationary_distr[1:3])
marginal_X1 = sum(joint_stationary_distr[4:6])
marginal_X2 = sum(joint_stationary_distr[7:9])
marginal_X3 = sum(joint_stationary_distr[10:12])
marginal_X4 = sum(joint_stationary_distr[13:15])
marginal_X5 = sum(joint_stationary_distr[16:18])

marginalX = vcat(marginal_X0, marginal_X1, marginal_X2, marginal_X3, marginal_X4, marginal_X5)
#stationary marginal distribution of X

#4.3
X_values = collect(0:5)

EX = marginalX' * X_values #EX = 0.7098214285714288
