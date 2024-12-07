
using NLsolve

function internal_rate(C::Vector{Float64})
    function NPV(r::Float64, C::Vector{Float64})
        discount_vector = [] 
        for i in 1:length(C)
            discount_factor = (1.0+r)^-(i-1.0)
            push!(discount_vector, discount_factor)
        end
        return discount_vector'*C
    end
    function wrapped_NPV(r::Vector{Float64})
        return [NPV(r[1], C)]
    end
    initial_guess = [0.1]
    IRR = nlsolve(wrapped_NPV, initial_guess; ftol=1e-14, show_trace=true);
    if IRR.f_converged == false || IRR.zero[1] <= -1.0
        println("WARNING: the solver did not find a valid solution IRR")
    else
        return IRR.zero[1]
    end
end

internal_rate([-5,0,0,2.5,5])

internal_rate([-1,1.1])

internal_rate([1,1.1])
internal_rate([-1,-1.1])

internal_rate([1.0,1.0])

internal_rate([-3.0,4.5,4.5,4.5,4.5,4.5,4.5])

#####################################################################
#NOTES

#based on nonlinear_1_jh.jl

#ALWAYS VERIFY THAT THIS IS THE CASE: "Converge: true";

using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, NLsolve, Roots


function NPV(r::Float64, C::Vector{Float64})
    discount_vector = []
    for i in 1:length(C)
        discount_factor = (1 + r)^-(i - 1)
        push!(discount_vector, discount_factor)
    end
    return sum(discount_vector .* C)
end

NPV(0.1,[-1,1.1])

NPV(0.11735,[-5,0,0,2.5,5])


# Wrapper to work with nlsolve
function wrapped_NPV(r::Vector{Float64})
    return NPV(r[1], [-5, 0, 0, 2.5, 5])
end

# Initial guess for the root
guess = [0.1]

# Solve using nlsolve
IRR = nlsolve(wrapped_NPV, guess; ftol=1e-14, show_trace=true)

IRR.zero[1]

