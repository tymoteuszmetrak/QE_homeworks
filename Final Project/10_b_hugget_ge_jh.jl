#Hugget General equilibrium (prices r,w determined endogeneously)


# load some packages we will need today

using Distributions, QuantEcon, IterTools, Optim, Interpolations, LinearAlgebra, Inequality, Statistics, ColorSchemes,PrettyTables, Plots, Parameters, Roots



@with_kw struct HAProblem

    ρ_z=0.96 # log of productivity persistence
    ν_z=sqrt(0.05) # volatility log of productivity 
    γ = 2 # curvature parameter of utility function
    u = γ == 1 ? x -> log(x) : x -> (x^(1 - γ) - 1) / (1 - γ) # utility function
    ϕ = 0.0 # borrowing constraint
    β = 0.96 # discount factor
    N_z = 5 # grid size for Tauchen
    mc_z = tauchen(N_z, ρ_z, ν_z, 0)
    λ_z = stationary_distributions(mc_z)[1]
    P_z = mc_z.p # transition matrix

    
    z_vec = exp.(mc_z.state_values) / sum(exp.(mc_z.state_values) .* λ_z) # normalize so that mean is 1

    a_max  = 150 # maximum assets
    N_a    = 150 # assets grid 

   
    a_min =  -ϕ # minimum assets

    rescaler = range(0,1,length=N_a) .^ 5.0 
    a_vec = a_min  .+ rescaler * (a_max - a_min) # grid for assets


   # a_vec =  collect(range(a_min, a_max, length=N_a))  # note - uniform grid here, not the best choice
end


function Tσ_operator(v,σ_ind,model,prices)

    @unpack  N_z, z_vec, P_z, β, a_vec, N_a, u = model
    @unpack  r, w = prices
    v_new = similar(v)
    for (z_ind, z) in enumerate(z_vec) # loop over productivity
        for (a_ind, a) in enumerate(a_vec) # loop over assets today
            #for (a_next_ind, a_next) in enumerate(a_vec) # loop over assets tomorrow
    
            a_next_ind  = σ_ind[a_ind,z_ind]    
            a_next      = a_vec[a_next_ind]
            v_new[a_ind,z_ind]   = u((1+r)*a + w*z - a_next) + β * sum( v[a_next_ind,z_next_ind] * P_z[z_ind,z_next_ind] for z_next_ind in 1:N_z )
        end
    end
           
    return v_new

end

function T_operator(v,model,prices)

    @unpack  N_z, z_vec, P_z, β, a_vec, N_a, u = model
    @unpack  r, w  = prices
    v_new   = zeros(Float64,N_a,N_z)
    σ       = zeros(Float64,N_a,N_z)
    σ_ind   = ones(Int,N_a,N_z)

    for (z_ind, z) in enumerate(z_vec) # loop over productivity
        for (a_ind, a) in enumerate(a_vec) # loop over assets today
            
            reward = zeros(N_a)

            for (a_next_ind, a_next) in enumerate(a_vec) # loop over assets tomorrow
                c = (1+r)*a + w*z - a_next
                util = c > 0 ? u(c) : -Inf
                reward[a_next_ind]   = util + β * sum( v[a_next_ind,z_next_ind] * P_z[z_ind,z_next_ind] for z_next_ind in 1:N_z )
            end

            v_new[a_ind,z_ind], σ_ind[a_ind,z_ind] = findmax(reward) # for each k, find the maximum reward and the optimal next level of capital
           
            σ[a_ind,z_ind] = a_vec[σ_ind[a_ind,z_ind]] # store the optimal policy 
        end
    end
    return v_new, σ,  σ_ind

end

function opi(model, prices; tol = 1e-8, maxiter = 1000, max_m = 1)
    error = tol + 1.0; iter = 1 #  initialize

    @unpack N_a, N_z = model
    v = zeros(N_a,N_z); 

    while error > tol && iter < maxiter
        v_new, σ_new, σ_ind_new = T_operator(v,model,prices)

        for m in 1:max_m
            v_new = Tσ_operator(v_new,σ_ind_new,model,prices)
        end
        error = maximum(abs.(v_new .- v))
        v = v_new
        iter += 1
    end
    # one more iteration to get the policy function
    v, σ, σ_ind = T_operator(v,model,prices)
    return v, σ, σ_ind, iter, error
        
end


prices = (r = 0.01, w = 1.0)
model = HAProblem(ϕ = 0.1)


v_opi, σ_opi, σ_ind_opi, iter_opi, error_opi = opi(model, prices,maxiter =5000, tol = 1e-11, max_m = 10)



lines_scheme = [get(ColorSchemes.thermal,LinRange(0.2,0.8,model.N_z));];
policy_plot = plot(xlabel = "a", ylabel = "a′", title = "Policy function");

for j in 1:model.N_z
    plot!(policy_plot,model.a_vec[1:75], σ_opi[1:75,j], label = false, color = lines_scheme[j], lw=3)
end

plot!(policy_plot,model.a_vec[1:75], model.a_vec[1:75], label = false, linestyle = :dash, color = :black)


value_plot = plot(xlabel = "a", ylabel = "V", title = "Value function");
for j in 1:model.N_z
    plot!(value_plot,model.a_vec[1:75], v_opi[1:75,j], label = false, color = lines_scheme[j], lw=3)
end




function get_transition(model, σ_ind)
    
        @unpack N_a, N_z, P_z, a_vec, z_vec = model
        
        Q = zeros(N_a * N_z,N_a * N_z)
        
        # could be done in a more pretty way
        for (z_ind, z) in enumerate(z_vec)
                for (z_next_ind, z′) in enumerate(z_vec)
                        Q[(z_ind-1)*N_a+1:z_ind*N_a,(z_next_ind-1)*(N_a)+1:z_next_ind*N_a] = (σ_ind[:,z_ind] .== (1:N_a)') * P_z[z_ind,z_next_ind]
                end
        end
        
    return Q
end

Q = get_transition(model, σ_ind_opi)

function stationary_distribution_hh(model, σ_ind)

    Q = get_transition(model, σ_ind)

    @unpack N_a, N_z, z_vec = model

    λ_vector = (Q^10000)[1,:]
    λ = zeros(N_a, N_z)

    for (j, z) in enumerate(z_vec)
        for (j, z′) in enumerate(z_vec)
            λ[:,j] = λ_vector[(j-1)*N_a+1:j*N_a]
        end
    end

    λ_a = sum(λ,dims=2)
    λ_z = sum(λ,dims=1)'
    return λ, λ_vector, λ_a, λ_z
end

λ, λ_vector, λ_a, λ_z = stationary_distribution_hh(model, σ_ind_opi)


p1 = plot(model.a_vec, λ_a, xlabel = "a", ylabel = "λ(a)",title = "Assets", legend = false)
p2 = plot(model.z_vec, λ_z, xlabel = "z", ylabel = "λ(z)", title = "Productivity",  legend = false)
plot(p1,p2)


function solve_hh_block(model,prices)
    v_opi, σ_opi, σ_ind_opi, iter_opi, error_opi = opi(model, prices,maxiter =5000, tol = 1e-8, max_m = 10)
    λ, λ_vector, λ_a, λ_z = stationary_distribution_hh(model, σ_ind_opi)
    A′ = sum(λ .* σ_opi)

    return v_opi, σ_opi,σ_ind_opi,iter_opi,error_opi, λ, λ_vector, λ_a, λ_z, A′
end
        

v_opi, σ_opi,σ_ind_opi,iter_opi,error_opi, λ, λ_vector, λ_a, λ_z, A′ = solve_hh_block(model,prices)


function asset_demand_hugget(model,r)
    w = 1
    prices = (r = r,w = w)
    v_opi, σ_opi,σ_ind_opi,iter_opi,error_opi, λ, λ_vector, λ_a, λ_z, A′ = solve_hh_block(model,prices)
    
    #display("value function iteration error = $error_opi")

    return A′
end
   
asset_demand_hugget(model,0.01)


grid_r = LinRange(-0.2,model.β^(-1)-1-0.03,15)
asset_demand = zeros(length(grid_r))
for (ind_r,r) in enumerate(grid_r)
    A′ = asset_demand_hugget(model,r)
    asset_demand[ind_r] = A′
end


plot(asset_demand,grid_r,label="Asset Demand",ylabel="Rate",xlabel="Assets",legend=:bottomright,linewidth=3)
hline!([model.β^(-1)-1],label="r=β^(-1)-1",linewidth=3,linestyle=:dash,color=:black)



r_eqm = find_zero(x -> asset_demand_hugget(model,x), (-0.05, 0.0 ), verbose = true, maxiter = 10)
display("Equilibrium interest rate is $r_eqm")
display("Aggregate excess asset demand $(asset_demand_hugget(model,r_eqm))") # note - does not clear fully because of the grid 