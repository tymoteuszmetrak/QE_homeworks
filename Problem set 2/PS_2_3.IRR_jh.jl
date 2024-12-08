###Problem 3: Internal rate of return 

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
    wrapped_NPV(r) = [NPV(r[1], C)] #r as a vector to accommodate nlsolve algorithm
    initial_guess = [0.1]
    IRR = nlsolve(wrapped_NPV, initial_guess; ftol=1e-14, show_trace=true);
    if IRR.f_converged == false || IRR.zero[1] <= -1.0
        println("WARNING: the solver did not find a valid IRR")
    else
        return IRR.zero[1]
    end
end

internal_rate([-3.0,4.5,4.5,4.5,4.5,4.5,4.5])

internal_rate([-5,0,0,2.5,5])

internal_rate([-1,1.1])

internal_rate([1,1.1])
internal_rate([-1,-1.1])
internal_rate([1.0,1.0])