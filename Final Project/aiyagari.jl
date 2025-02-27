

using Distributions, QuantEcon, IterTools, Plots, Optim, Interpolations, LinearAlgebra, Inequality, Statistics, ColorSchemes,PrettyTables, Roots, Revise, Parameters

### LOAD MODULE 
includet("aiyagari_module.jl")
using .Aiyagari

# access to all exported functions
hh = HAProblem()

# try to see if it works as intended
prices = (r=0.01, w=1.0)
taxes  = (τ_w = 0.3, τ_r = 0.0, d = 0.0)


v_opi, σ_opi,σ_ind_opi,iter_opi,error_opi, λ, λ_vector, λ_a, λ_z, A′ = solve_hh_block(hh,prices,taxes)


println("error in VFI = $error_opi")


# test plotting
    lines_scheme = [get(ColorSchemes.thermal,LinRange(0.0,1.0,hh.N_z));];
    value_plot = plot(xlabel = "a", ylabel = "V", title = "Value function");


    for j in 1:hh.N_z
        plot!(hh.a_vec, v_opi[:,j], label = false, color = lines_scheme[j], lw=3)
    end
    value_plot

# now do the example of the full thing

    firm = FIRM()
    govt = GOVT(τ_r = 0.0, τ_w = 0.0, d = 0.0, B = 0.0)
    L = get_aggregate_labor(hh)

    r_init = 0.0
    K_L, w = solve_firm(firm,r_init)
    K = K_L * L
    asset_supply = K + govt.B
    
    prices = (r=r_init, w=w)
    taxes = (τ_w = govt.τ_w, τ_r = govt.τ_r, d = govt.d)

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
        asset_supply = K + govt.B
        prices = (r=r, w=w)
        taxes = (τ_w = govt.τ_w, τ_r = govt.τ_r, d = govt.d)
        residual = asset_demand(hh,prices,taxes) - asset_supply

        return residual
    end

# test 

    r_guess = 0.01
    aiyagari_residual(r_guess,hh,govt,firm)

# plot 


    grid_r = LinRange(-0.02,0.00,15)
    excess_asset_demand = zeros(length(grid_r))
    for (ind_r,r_guess) in enumerate(grid_r)
        excess = aiyagari_residual(r_guess,hh,govt,firm)
        excess_asset_demand[ind_r] = excess
    end
    plot(excess_asset_demand,grid_r,label="Excess Asset Demand",ylabel="Rate",xlabel="Assets",legend=:bottomright,linewidth=3)


# solve for an equilibrium return 
    r_star = find_zero(x -> aiyagari_residual(x,hh,govt,firm), (-0.02, 0.0 ) ,verbose = true, maxiter = 10)
    println("equilibrium real rate = $r_star")
    display("Aggregate excess asset demand $(aiyagari_residual(r_star,hh,govt,firm))") # note - does not clear fully because of the grid 
 