
#Problem 3: Convergence in the Neoclassical Growth Model

## Neoclassical growth model
using  Plots, Parameters, PrettyTables


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