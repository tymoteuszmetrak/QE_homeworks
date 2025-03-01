
using Distributions, QuantEcon, IterTools, Plots, Optim, Interpolations, LinearAlgebra, Inequality, Statistics, ColorSchemes,PrettyTables, Roots, Revise, Parameters
using Inequality, LaTeXStrings, BenchmarkTools, LoopVectorization, NLsolve

#1 - (2/3) * sum(hh.λ_z .* hh.z_vec.^0.85)
##################################################################################

guess = 0.9340109



B_star = nlsolve(residual_beta,[guess],ftol=1e-1,show_trace=true)

β_star_2 = find_zero(residual_beta, 0.935, verbose=true)
println("Equilibrium β = $β_star")

B_star_3 = nlsolve(residual_beta, [guess], method=:trust_region, ftol=1e-3, show_trace=true)





#########################################

# Verify excess demand at the equilibrium β
excess_demand = aiyagari_residual_beta(β_star, hh, govt, firm, 0.04, 1.0)
println("Excess demand at equilibrium β: $excess_demand")

# Update the household problem with the equilibrium β
hh = HAProblem(β=β_star)

# Solve the model with the updated β
prices = (r=0.04, w=1.0)
taxes = (τ_w=govt.τ_w, lambda=govt.lambda)
v_opi, σ_opi, σ_ind_opi, iter_opi, error_opi, λ, λ_vector, λ_a, λ_z, A′ = solve_hh_block(hh, prices, taxes)

# Calculate asset supply (K)
L = get_aggregate_labor(hh)
K_L, _ = solve_firm(firm, 0.04)
K = K_L * L
asset_supply = K

# Calculate excess demand
excess_demand = A′ - asset_supply
println("Final excess demand: $excess_demand")


###########################################################


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

excess_demand = asset_demand(hh,prices,taxes) - asset_supply #DO POPRAWY 141 wychodzi - to dużo, tak ma być?

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