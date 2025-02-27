
### MODULE 
module Aiyagari
using Distributions, QuantEcon, IterTools, Plots, Optim, Interpolations, LinearAlgebra, Inequality, Statistics, ColorSchemes,PrettyTables, Roots, Revise, Parameters

export FIRM, GOVT, HAProblem
export Tσ_operator, T_operator, opi
export get_transition, stationary_distribution_hh, solve_hh_block, get_aggregate_labor, solve_firm




@with_kw struct FIRM
    α = 1/3 # capital share
    δ = 0.1 # depreciation rate
    Z = 1 # productivity
    F   =  (K,L) ->  Z * K^α * L^(1-α) # production function
    F_K =  (K,L) ->  α * Z * K^(α-1) * L^(1-α) # marginal product of capital
    F_L =  (K,L) -> (1-α) * Z * K^α * L^(-α) # marginal product of labor
end



function solve_firm(firm,r)
    @unpack α, δ, F, F_K, F_L = firm
    K_L = (α/(r+δ))^(1/(1-α)) # capital to output ratio
    w = F_L(K_L,1) # wage
    return K_L, w
end

function get_aggregate_labor(ha_block)
    @unpack N_z, z_vec, λ_z = ha_block
    L =  sum(z_vec .* λ_z)
    return L
end


@with_kw struct GOVT
    τ_w = 0.1 # labor tax
    τ_r = 0.1 # capital tax
    d = 0.0 # lump-sum transfer
    B = 0.0 # debt
end

function get_G(L, A, prices, government)
    @unpack r, w = prices
    @unpack τ_w, τ_r, d, B = government
    G = τ_w * w * L + τ_r * r * A - d - r * B
    return G
end



@with_kw struct HAProblem

    ρ_z=0.96 # log of productivity persistence
    ν_z=sqrt(0.125) # volatility log of productivity 
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



function Tσ_operator(v,σ_ind,model,prices,taxes )

    @unpack  N_z, z_vec, P_z, β, a_vec, N_a, u = model
    @unpack  r, w = prices
    @unpack τ_w, τ_r, d = taxes
    v_new = similar(v)
    for (z_ind, z) in enumerate(z_vec) # loop over productivity
        for (a_ind, a) in enumerate(a_vec) # loop over assets today
            #for (a_next_ind, a_next) in enumerate(a_vec) # loop over assets tomorrow
    
            a_next_ind  = σ_ind[a_ind,z_ind]    
            a_next      = a_vec[a_next_ind]
            v_new[a_ind,z_ind]   = u((1-τ_r)*(1+r)*a + (1-τ_w)*w*z - a_next + d) + β * sum( v[a_next_ind,z_next_ind] * P_z[z_ind,z_next_ind] for z_next_ind in 1:N_z )
        end
    end
           
    return v_new

end

function T_operator(v,model,prices, taxes)

    @unpack  N_z, z_vec, P_z, β, a_vec, N_a, u = model
    @unpack  r, w  = prices
    @unpack τ_w, τ_r, d = taxes
    v_new   = zeros(Float64,N_a,N_z)
    σ       = zeros(Float64,N_a,N_z)
    σ_ind   = ones(Int,N_a,N_z)

    for (z_ind, z) in enumerate(z_vec) # loop over productivity
        for (a_ind, a) in enumerate(a_vec) # loop over assets today
            
            reward = zeros(N_a)

            for (a_next_ind, a_next) in enumerate(a_vec) # loop over assets tomorrow
                c = (1-τ_r)*(1+r)*a + (1-τ_w)*w*z + d - a_next
                util = c > 0 ? u(c) : -Inf
                reward[a_next_ind]   = util + β * sum( v[a_next_ind,z_next_ind] * P_z[z_ind,z_next_ind] for z_next_ind in 1:N_z )
            end

            v_new[a_ind,z_ind], σ_ind[a_ind,z_ind] = findmax(reward) # for each k, find the maximum reward and the optimal next level of capital
           
            σ[a_ind,z_ind] = a_vec[σ_ind[a_ind,z_ind]] # store the optimal policy 
        end
    end
    return v_new, σ,  σ_ind

end

function opi(model, prices, taxes; tol = 1e-8, maxiter = 1000, max_m = 1)
    error = tol + 1.0; iter = 1 #  initialize

    @unpack N_a, N_z = model
    v = zeros(N_a,N_z); 

    while error > tol && iter < maxiter
        v_new, σ_new, σ_ind_new = T_operator(v,model,prices,taxes)

        for m in 1:max_m
            v_new = Tσ_operator(v_new,σ_ind_new,model,prices,taxes)
        end
        error = maximum(abs.(v_new .- v))
        v = v_new
        iter += 1
    end
    # one more iteration to get the policy function
    v, σ, σ_ind = T_operator(v,model,prices,taxes)
    return v, σ, σ_ind, iter, error
        
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

function solve_hh_block(model, prices, taxes)
    v_opi, σ_opi, σ_ind_opi, iter_opi, error_opi = opi(model, prices, taxes,maxiter =5000, tol = 1e-8, max_m = 10)
    λ, λ_vector, λ_a, λ_z = stationary_distribution_hh(model, σ_ind_opi)
    A′ = sum(λ .* σ_opi)

    return v_opi, σ_opi,σ_ind_opi,iter_opi,error_opi, λ, λ_vector, λ_a, λ_z, A′
end


function show_statistics(ha_block,grid,λ_a,λ_z)
# warning - this can be misleading if we allow for negative values!
lorenz_a_pop,lorenz_a_share=lorenz_curve(grid.a_vec,vec(λ_a))
lorenz_z_pop,lorenz_z_share=lorenz_curve(ha_block.z_vec,vec(λ_z))



lorenz_a = LinearInterpolation(lorenz_a_pop, lorenz_a_share);
lorenz_z = LinearInterpolation(lorenz_z_pop, lorenz_z_share);


header = (["", "Assets", "Income"])

data = [           
                     "Bottom 50% share"         lorenz_a(0.5)        lorenz_z(0.5)    ;
                     "Top 10% share"            1-lorenz_a(0.9)         1-lorenz_z(0.9)     ;
                     "Top 1% share"             1-lorenz_a(0.99)        1-lorenz_z(0.99)    ;  
                     "Gini Coefficient"      wgini(grid.a_vec,vec(max.(0,λ_a)))      wgini(ha_block.z_vec,vec(max.(0.0,λ_z)))    ;]

return pretty_table(data;header=header,formatters=ft_printf("%5.3f",2:3))
end

function solve_hh_block(model,prices)
    v_opi, σ_opi, σ_ind_opi, iter_opi, error_opi = opi(model, prices,maxiter =5000, tol = 1e-11, max_m = 10)
    λ, λ_vector, λ_a, λ_z = stationary_distribution_hh(model, σ_ind_opi)
    A′ = sum(λ .* σ_opi)

    return v_opi, σ_opi,σ_ind_opi,iter_opi,error_opi, λ, λ_vector, λ_a, λ_z, A′
end


end # module ends

