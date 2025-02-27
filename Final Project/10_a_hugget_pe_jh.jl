#Hugget Partial equilibrium: hholds optimize and take prices r,w as given (determined exogeneously)


# load some packages we will need today

using Distributions, QuantEcon, IterTools, Optim, Interpolations, LinearAlgebra, Inequality, Statistics, ColorSchemes,PrettyTables, Plots, Parameters


@with_kw struct HAProblem

    ρ_z=0.96 # log of productivity persistence
    ν_z=sqrt(0.125) # volatility log of productivity
    γ = 2 # curvature parameter of utility function
    u = γ == 1 ? x -> log(x) : x -> (x^(1 - γ) - 1) / (1 - γ) # utility function
    ϕ = 0.0 # borrowing constraint
    β = 0.96 # discount factor
    N_z = 5 # grid size for Tauchen
    mc_z = tauchen(N_z, ρ_z, ν_z, 0)
    λ_z = stationary_distributions(mc_z)[1] #Quant Econ function; sum up to 1
    P_z = mc_z.p # transition matrix

    
    z_vec = exp.(mc_z.state_values) / sum(exp.(mc_z.state_values) .* λ_z) # normalize so that mean is 1?

    a_max  = 150 # maximum assets
    N_a    = 150 # assets grid 

   
    a_min =  -ϕ # minimum assets

    rescaler = range(0,1,length=N_a) .^ 5.0 #150 numbers from 0 to 1 to ^5
    a_vec = a_min  .+ rescaler * (a_max - a_min) # grid for assets
#more pts near 0 very quickly and big gaps for largest levels - ln is concave, we need many pts at the steep ln part to capture value function precisely

   # a_vec =  collect(range(a_min, a_max, length=N_a))  # note - uniform grid here, not the best choice
end


function Tσ_operator(v,σ_ind,model,prices) #corresponds to OPI, only for one a'

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

function T_operator(v,model,prices) #full vfi

    @unpack  N_z, z_vec, P_z, β, a_vec, N_a, u = model
    @unpack  r, w  = prices
    v_new   = zeros(Float64,N_a,N_z)
    σ       = zeros(Float64,N_a,N_z)
    σ_ind   = ones(Int,N_a,N_z) #will store individual policies based on a,z

    for (z_ind, z) in enumerate(z_vec) # loop over productivity today
        for (a_ind, a) in enumerate(a_vec) # loop over assets today
            
            reward = zeros(N_a) #length 150

            for (a_next_ind, a_next) in enumerate(a_vec) # loop over assets tomorrow; no loop over z' because we don't know it and can't choose it
                c = (1+r)*a + w*z - a_next
                util = c > 0 ? u(c) : -Inf
                reward[a_next_ind]   = util + β * sum( v[a_next_ind,z_next_ind] * P_z[z_ind,z_next_ind] for z_next_ind in 1:N_z ) #unoptimized v(a')
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
model = HAProblem()


v_opi, σ_opi, σ_ind_opi, iter_opi, error_opi = opi(model, prices,maxiter =5000, tol = 1e-11, max_m = 10)

error_opi #very small difference between consecutive v iterations, good
iter_opi #70; not bad

lines_scheme = [get(ColorSchemes.thermal,LinRange(0.2,0.8,model.N_z));]
policy_plot = plot(xlabel = "a", ylabel = "a′", title = "Policy function");
#a'>a if above 45 degree line

for j in 1:model.N_z
    plot!(policy_plot,model.a_vec[1:75], σ_opi[1:75,j], label = false, color = lines_scheme[j], lw=3)
end

plot!(policy_plot,model.a_vec[1:75], model.a_vec[1:75], label = false, linestyle = :dash, color = :black)
#a' above 

value_plot = plot(xlabel = "a", ylabel = "V", title = "Value function")
for j in 1:model.N_z
    plot!(value_plot,model.a_vec[1:75], v_opi[1:75,j], label = false, color = lines_scheme[j], lw=3)
end

#for much lower persistance of z, most people will find themselves every period in mode value of z
#0.1 -> #nice monotonicity - v lines corresponding to different z won't touch each other
#a is very concave only for low z; with high z you dont want to consume everything, you want too save


function get_transition(model, σ_ind)
    
        @unpack N_a, N_z, P_z, a_vec, z_vec = model
        
        Q = zeros(N_a * N_z,N_a * N_z) #number of a,z pairs (150*5) -> 750x750
        
        # could be done in a more pretty way
        for (z_ind, z) in enumerate(z_vec) #fix level of z
                for (z_next_ind, z′) in enumerate(z_vec) #fix level of a tomorrow
                        Q[(z_ind-1)*N_a+1:z_ind*N_a,(z_next_ind-1)*(N_a)+1:z_next_ind*N_a] = (σ_ind[:,z_ind] .== (1:N_a)') * P_z[z_ind,z_next_ind]
                                                                                                                              #P(z,z')
                end
        end
        
    return Q
end

#investigating the function

z_ind = 2
z_next_ind = 3
N_a = 150
σ_ind = σ_ind_opi 
P_z[z_ind,z_next_ind]
(z_ind-1)*N_a+1:z_ind*N_a,(z_next_ind-1)*(N_a)+1:z_next_ind*N_a #(151:300, 301:450)
#5 by 5 blocks (for z and z' pairs); inside a block: 150 rows and columns (N_a = 150)
σ_ind[:,z_next_ind].==(1:N_a)'


Q = get_transition(model, σ_ind_opi)
surface(Q) #probability of transitions

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


lorenz_a_pop,lorenz_a_share=lorenz_curve(model.a_vec,vec(λ_a))
plot(lorenz_a_pop,lorenz_a_share,xlabel="Cumulative share of population",ylabel="Cumulative share of assets",title="Lorenz curve",legend=false)
plot!(lorenz_a_pop,lorenz_a_pop,linestyle=:dash,color=:black)

function show_statistics_hugget(model,λ_a,λ_z)

    @unpack a_vec, z_vec = model
    # warning - this can be misleading if we allow for negative values!
    lorenz_a_pop,lorenz_a_share=lorenz_curve(a_vec,vec(λ_a))
    lorenz_z_pop,lorenz_z_share=lorenz_curve(z_vec,vec(λ_z))
    
    
    
    lorenz_a = LinearInterpolation(lorenz_a_pop, lorenz_a_share);
    lorenz_z = LinearInterpolation(lorenz_z_pop, lorenz_z_share);
    
    
    header = (["", "Assets", "Income"])
    
    data = [           
                         "Bottom 50% share"         lorenz_a(0.5)        lorenz_z(0.5)    ;
                         "Top 10% share"            1-lorenz_a(0.9)         1-lorenz_z(0.9)     ;
                         "Top 1% share"             1-lorenz_a(0.99)        1-lorenz_z(0.99)    ;  
                         "Gini Coefficient"      wgini(a_vec,vec(max.(0,λ_a)))      wgini(z_vec,vec(max.(0.0,λ_z)))    ;]
    
    return pretty_table(data;header=header,formatters=ft_printf("%5.3f",2:3))
end
    
    
    show_statistics_hugget(model,λ_a,λ_z)