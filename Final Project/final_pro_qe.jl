# Wojciech Szymczak, Jan Hincz, Tymoteusz Metrak
# Quantitative Economics Final Project 
# February 2025

# Acknowledgement: We thank Piot Zoch and Marcin Lewandowski for class materials used within thins project

# Read packages needed for the analysis 
using Distributions, QuantEcon, IterTools, Plots, Optim, Interpolations, LinearAlgebra, Inequality, Statistics, ColorSchemes,PrettyTables, Roots, Revise, Parameters
using Inequality, LaTeXStrings, BenchmarkTools, LoopVectorization, NLsolve

# Set up parameters and function (do not change with lambda )
@with_kw struct FIRM
    α = 0.4 # capital share (USA 2019 labor share = 0.6: Penn World Table 10.01)
    δ = 0.04 # depreciation rate
    Z = 0.7137 #TFP (in write-up: A; solution in write-up)
    F   =  (K,L) ->  Z * K^α * L^(1-α) # production function
    F_K =  (K,L) ->  α * Z * K^(α-1) * L^(1-α) # marginal product of capital
    F_L =  (K,L) -> (1-α) * Z * K^α * L^(-α) # marginal product of labor
end

# Further functions for solving 
function solve_firm(firm,r)
    @unpack α, δ, Z, F, F_K, F_L = firm
    K_L = ((α*Z)/(r+δ))^(1/(1-α)) # capital to output ratio
    w = F_L(K_L,1) # wage, L normalized to 1
    return K_L, w
end

# Labor function 
function get_aggregate_labor(ha_block)
    @unpack N_z, z_vec, λ_z = ha_block
    L =  sum(z_vec .* λ_z)
    return L
end

# gov parameters when lambda = 0.0
@with_kw struct GOVT
    τ_w = 1/3 
    lambda = 0.0    
end


# parameters for household problem 
@with_kw struct HAProblem

    ρ_z = 0.9 # persistence of log productivity (in write-up := ρ)
    ν_z = 0.4 # std of log productivity (in write-up := σ)
    ln_z_tilde = -8.0/19.0 #z_tilde = e^(-8/19) := constant normalizing average productivity to 1
    γ = 2 # curvature parameter of utility function
    u = γ == 1 ? x -> log(x) : x -> (x^(1 - γ) - 1) / (1 - γ) # utility function
    ϕ = 0.0 # borrowing constraint
    β = 0.9340109 #already optimal parameter placed here 
    N_z = 5 # grid size for Tauchen
    mc_z = tauchen(N_z, ρ_z, ν_z, ln_z_tilde)
    λ_z = stationary_distributions(mc_z)[1]
    P_z = mc_z.p # transition matrix

    z_vec = exp.(mc_z.state_values) / sum(exp.(mc_z.state_values) .* λ_z) # normalize so that mean is 1

    a_max  = 25 # 
    N_a    = 300 # assets grid 

   
    a_min =  -ϕ # minimum assets
    

    rescaler = range(0,1,length=N_a) .^ 5.0

    a_vec = a_min  .+ rescaler * (a_max - a_min) # grid for assets


   # a_vec =  collect(range(a_min, a_max, length=N_a))  # note - uniform grid here, not the best choice
end



function Tσ_operator(v,σ_ind,model,prices,taxes)

    @unpack  N_z, z_vec, P_z, β, a_vec, N_a, u = model
    @unpack  r, w = prices
    @unpack τ_w, lambda = taxes
    v_new = similar(v)
    for (z_ind, z) in enumerate(z_vec) # loop over productivity
        for (a_ind, a) in enumerate(a_vec) # loop over assets today
            #for (a_next_ind, a_next) in enumerate(a_vec) # loop over assets tomorrow
    
            a_next_ind  = σ_ind[a_ind,z_ind]    
            a_next      = a_vec[a_next_ind]
            v_new[a_ind,z_ind]   = u((1+r)*a + (1-τ_w)*w*(z^(1-lambda)) - a_next) + β * sum( v[a_next_ind,z_next_ind] * P_z[z_ind,z_next_ind] for z_next_ind in 1:N_z ) #DO POPRAWY - ŻEBY PASOWAŁO TEŻ DLA lambda = 0.15
        end
    end
           
    return v_new

end

function T_operator(v,model,prices,taxes)

    @unpack  N_z, z_vec, P_z, β, a_vec, N_a, u = model
    @unpack  r, w  = prices
    @unpack τ_w, lambda = taxes
    v_new   = zeros(Float64,N_a,N_z)
    σ       = zeros(Float64,N_a,N_z)
    σ_ind   = ones(Int,N_a,N_z)

    for (z_ind, z) in enumerate(z_vec) # loop over productivity
        for (a_ind, a) in enumerate(a_vec) # loop over assets today
            
            reward = zeros(N_a)

            for (a_next_ind, a_next) in enumerate(a_vec) # loop over assets tomorrow
                c = (1+r)*a + (1-τ_w)*w*(z^(1-lambda)) - a_next
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


# access to all exported functions
hh = HAProblem()

# try to see if it works as intended
prices = (r=0.04, w=1.0)
taxes  = (τ_w = 1/3, lambda = 0.0)


v_opi, σ_opi,σ_ind_opi,iter_opi,error_opi, λ, λ_vector, λ_a, λ_z, A′ = solve_hh_block(hh,prices,taxes)


println("error in VFI = $error_opi")


# Add the new function for β search
function aiyagari_residual_beta(β, hh, govt, firm, r, w)
    hh = HAProblem(β=β[1])  # Update β in the household problem
    prices = (r=r, w=w)
    taxes = (τ_w=govt.τ_w, lambda=govt.lambda)
    v_opi, σ_opi, σ_ind_opi, iter_opi, error_opi, λ, λ_vector, λ_a, λ_z, A′ = solve_hh_block(hh, prices, taxes)
    L = get_aggregate_labor(hh)
    K_L, _ = solve_firm(firm, r)
    K = K_L * L
    asset_supply = K
    excess_demand = A′ - asset_supply
    return excess_demand
end

# Search for β that clears the asset market
residual_beta(β) = aiyagari_residual_beta(β, hh, govt, firm, 0.04, 1.0)
firm = FIRM()
govt = GOVT(τ_w = 1/3, lambda = 0.0)
residual_beta(0.9340109) #0.00467227784140789, found by trial and error, no tested findzero and nsolve algos did not work properly

# NOTE THIS WAS CALCULATED AND BETA CALLIBRATED HERE 

##################################################################################

#############################REMAINING PARTS FOR lambda = 0.15

#FINDING tax rate for lambda = 0.15 
τ_w2 = 1 - (2/3) * sum(hh.λ_z .* hh.z_vec.^0.85) #0.85 = 1 - 0.15; G/Y at constant 0.2


#finding equilibrium prices for prices for lambda = 0.15 and corresponding τ_w2
hh2 = HAProblem(β = 0.9340109) #updated Beta, which sets excess demand close to 0
taxes2  = (τ_w = τ_w2, lambda = 0.15)

################################################3
# test plotting
    lines_scheme = [get(ColorSchemes.thermal,LinRange(0.0,1.0,hh.N_z));];
    value_plot = plot(xlabel = "a", ylabel = "V", title = "Value function");


    for j in 1:hh.N_z
        plot!(hh.a_vec, v_opi[:,j], label = false, color = lines_scheme[j], lw=3)
    end
    value_plot

# now do the example of the full thing

    firm = FIRM()
    govt = GOVT(τ_w = 1/3, lambda = 0.0)
    L = get_aggregate_labor(hh)

    r_init = 0.04
    K_L, w = solve_firm(firm,r_init)
    K = K_L * L
    asset_supply = K
    
    prices = (r=r_init, w=w)
    taxes = (τ_w = govt.τ_w, lambda = govt.lambda)

    function asset_demand(hh,prices,taxes)
        v_opi, σ_opi,σ_ind_opi,iter_opi,error_opi, λ, λ_vector, λ_a, λ_z, A′  = solve_hh_block(hh,prices, taxes)
        return A′
    end

    asset_demand(hh,prices,taxes)

    excess_demand = asset_demand(hh,prices,taxes) - asset_supply 
# put all pieces together
    function aiyagari_residual(r,hh,govt,firm)
        
        L = get_aggregate_labor(hh)
        K_L, w = solve_firm(firm,r)
        K = K_L * L
        asset_supply = K
        prices = (r=r, w=w)
        taxes = (τ_w = govt.τ_w, lambda = govt.lambda)
        residual = asset_demand(hh,prices,taxes) - asset_supply

        return residual
    end

# test 

    r_guess = 0.01
    aiyagari_residual(r_guess,hh,govt,firm)

# plot 


    grid_r = LinRange(-0.02,0.1,15)
    excess_asset_demand = zeros(length(grid_r))
    for (ind_r,r_guess) in enumerate(grid_r)
        excess = aiyagari_residual(r_guess,hh,govt,firm)
        excess_asset_demand[ind_r] = excess
    end
    plot(excess_asset_demand,grid_r,label="Excess Asset Demand",ylabel="Rate",xlabel="Assets",legend=:bottomright,linewidth=3)


# solve for an equilibrium return 
    r_star = find_zero(x -> aiyagari_residual(x,hh,govt,firm), (-0.02, 0.1) ,verbose = true, maxiter = 10)
    println("equilibrium real rate = $r_star")
    display("Aggregate excess asset demand $(aiyagari_residual(r_star,hh,govt,firm))") # note - does not clear fully because of the grid 
 
##################################################################################



# Generalize solving to compare economies
# 
function compute_equilibrium(firm, govt, hh, r, τ_w, lambda)
    K_L, w = solve_firm(firm, r)
    L = get_aggregate_labor(hh)
    K = K_L * L
    govt = GOVT(τ_w = τ_w, lambda = lambda)
    prices = (r=r, w=w)   
    
    v_opi, σ_opi, σ_ind_opi, iter_opi, error_opi, λ, λ_vector, λ_a, λ_z, A′ = solve_hh_block(hh, prices, (τ_w=govt.τ_w, lambda=govt.lambda))
    
    post_tax_income = (1 - govt.τ_w) .* (hh.z_vec .^ (1 - govt.lambda)) .* mean(hh.z_vec)
    gini_income = wgini(post_tax_income, vec(λ_z))
    gini_assets = wgini(hh.a_vec, vec(max.(0, λ_a)))
    
    return (r=r, w=w, tax_rate = govt.τ_w, K_to_output = K / (firm.Z * K^firm.α * L^(1 - firm.α)), 
            gini_income = gini_income, gini_assets = gini_assets,
            v_opi = v_opi, σ_opi = σ_opi, λ_a=λ_a, λ_z=λ_z)
end


# Solve for lambda = 0 and lambda = 0.15
# lambda = 0
firm = FIRM()
govt1 = GOVT(τ_w = 1/3, lambda = 0.0)
hh1 = HAProblem()  # Uses default parameters
r_star = find_zero(x -> aiyagari_residual(x, hh1, govt1, firm), (-0.02, 0.1), verbose = true, maxiter = 10) # find equilibrium rate 
# Compute equilibrium statistics for lambda = 0
eq1 = compute_equilibrium(firm, govt1, hh1, r_star, 1/3, 0.0)
σ_opi1 = eq1.σ_opi
v_opi1 = eq1.v_opi
λ_a1 = eq1.λ_a
λ_z1 = eq1.λ_z


# Recalculate the same for lambda = 0.15 
# WHAT CHANGES: taxes, wage, r
τ_w2 = 1 - (2/3) * sum(hh1.λ_z .* hh1.z_vec.^0.85) 
govt2 = GOVT(τ_w = τ_w2, lambda = 0.15)
hh2 = HAProblem()  # Using same beta as calibrated before
r_star2 = find_zero(x -> aiyagari_residual(x, hh2, govt2, firm), (-0.02, 0.1), verbose = true, maxiter = 10)
eq2 = compute_equilibrium(firm, govt2, hh2, r_star2, τ_w2, 0.15)
σ_opi2 = eq2.σ_opi
v_opi2 = eq2.v_opi
λ_a2 = eq2.λ_a
λ_z2 = eq2.λ_z

# Instead of pretty tables 
println("Comparing Equilibrium Outcomes:")
println("-------------------------------------------------")
println("Metric                 | Lambda = 0  | Lambda = 0.15")
println("-------------------------------------------------")
println("Interest Rate (r)      | $(eq1.r)  | $(eq2.r)")
println("Wage Rate (w)          | $(eq1.w)  | $(eq2.w)")
println("Tax Rate (τ_w)         | $(eq1.tax_rate) | $(eq2.tax_rate)")
println("K/Y Ratio              | $(eq1.K_to_output) | $(eq2.K_to_output)")
println("Gini Coefficient (Income) | $(eq1.gini_income) | $(eq2.gini_income)")
println("Gini Coefficient (Assets) | $(eq1.gini_assets) | $(eq2.gini_assets)")

#Comparing Equilibrium Outcomes:
#-------------------------------------------------
#Metric                 | Lambda = 0  | Lambda = 0.15
#-------------------------------------------------
#Interest Rate (r)      | 0.04000025212813271  | 0.046616594638326785
#Wage Rate (w)          | 0.999976595390981  | 0.9483820721991317
#Tax Rate (τ_w)         | 0.3333333333333333 | 0.3855118580576027
#K/Y Ratio              | 4.999984242041367 | 4.61805271461232
#Gini Coefficient (Income) | 0.5547279446668512 | 0.4858190957236648
#Gini Coefficient (Assets) | 0.46908277663292 | 0.4789807000142119


# Policy function - plotting - plot both economies for comparison
# lambda = 0.0
plot(hh1.a_vec, σ_opi1[:, 1], label="Policy λ=0 (low z)", lw=2)
plot!(hh1.a_vec, σ_opi1[:, end], label="Policy λ=0 (high z)", lw=2)
# lambda = 0.15
plot!(hh2.a_vec, σ_opi2[:, 1], label="Policy λ=0.15 (low z)", lw=2, linestyle=:dash)
plot!(hh2.a_vec, σ_opi2[:, end], label="Policy λ=0.15 (high z)", lw=2, linestyle=:dash)
title!("Policy Function Comparison")
xlabel!("Assets")
ylabel!("Next Period Assets")
display(plot!())

    
# Value function
plot(hh1.a_vec, v_opi1[:, 1], label="Value λ=0 (low z)", lw=2)
plot!(hh1.a_vec, v_opi1[:, end], label="Value λ=0 (high z)", lw=2)
plot!(hh2.a_vec, v_opi2[:, 1], label="Value λ=0.15 (low z)", lw=2, linestyle=:dash)
plot!(hh2.a_vec, v_opi2[:, end], label="Value λ=0.15 (high z)", lw=2, linestyle=:dash)

title!("Value Function Comparison")
xlabel!("Assets")
ylabel!("Value")
display(plot!())


    
# Asset distribution
histogram(hh1.a_vec, weights=vec(λ_a1), normalize=true, bins=50, label="Asset Distribution (λ = 0)", alpha=0.5)
histogram!(hh2.a_vec, weights=vec(λ_a2), normalize=true, bins=50, label="Asset Distribution (λ = 0.15)", alpha=0.5)

title!("Asset Distribution Comparison")
xlabel!("Assets")
ylabel!("Density")
display(plot!())

    
# Lorenz curves
lorenz_assets1 = lorenz_curve(hh1.a_vec, vec(λ_a1))  
lorenz_assets2 = lorenz_curve(hh2.a_vec, vec(λ_a2))  

lorenz_income1 = lorenz_curve(hh1.z_vec, vec(λ_z1))  
lorenz_income2 = lorenz_curve(hh2.z_vec, vec(λ_z2))  

plot(lorenz_assets1[1], lorenz_assets1[2], label="Assets Lorenz (λ=0)", lw=2)
plot!(lorenz_assets2[1], lorenz_assets2[2], label="Assets Lorenz (λ=0.15)", lw=2, linestyle=:dash)

plot!(lorenz_income1[1], lorenz_income1[2], label="Income Lorenz (λ=0)", lw=2)
plot!(lorenz_income2[1], lorenz_income2[2], label="Income Lorenz (λ=0.15)", lw=2, linestyle=:dash)

title!("Lorenz Curve Comparison")
xlabel!("Cumulative Population Share")
ylabel!("Cumulative Wealth/Income Share")
display(plot!())

################################ Code ends here