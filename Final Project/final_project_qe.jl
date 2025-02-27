# Quantitative Economics Final Project
# Jan Hincz, Tymoteusz Mętrak, Wojciech Szymczak

# Needed packages
using Inequality
using Distributions, Plots, LaTeXStrings, LinearAlgebra, BenchmarkTools
using LoopVectorization



# Calibration 
# External sources:
# Penn Table - Labor Share (Share of labour compensation in GDP at current prices): 0.6 (Penn Tables, 2019, US)

# α (1 - labor share) = 0.4
# γ = 2

# ρ = 0.9
# σ = 0.4
# Assign z in a way that average productivity is 1
# r = 0.04
# ϕ = 0
# λ (progressivity rate), changes from 0.0 to 0.15

# How to assign the rest of the variables?
# β (discounting factor) 
# τ (tax rate), 

# δ (depreciation),
# A (technology multiplier) 
# Wage rate w = 1
# Goverment spending to ouput = 0.2
# Investment to output = 0.2





# Hint for tax rate: for λ = 0, and wage 1 labor income is equal to 1, which allows to calculate product easier (knowing 1-alpha)
    # 1-alpha = 0.6; output 1/0.6 = 5/3; G = 0.2*output -> G = 1/3;  
        # for λ = 0; G = τ*y_dashed, but since y_dashed is 1 and G is 1/3, that implies tau = 1/3?

# L = 1, w=1, r = 0.04, investment output ratio 0.2   
    # This leads to:
    # δ = 0.04;
    # A ~= 0.70;
    # K = 25/3

# Function calculating the indicators comparing the economies with different λ

# Compare r
# Compare w
# Compare τ
# Compare the ratio of capital to output K

# Compare the Gini coefficient for after-tax labor income.
# The Gini coefficient for assets.


# Source: Julia Quant Economoics
function gini(v)
    (2 * sum(i * y for (i, y) in enumerate(v)) / sum(v)
     -
     (length(v) + 1)) / length(v)
end


# You also need to plot:
# 1. Value functions for both economies.
# 2. Policy functions for both economies.
# 3. The marginal distribution of assets for both economies.

# 4. The Lorenz curves for after-tax labor income and assets for both economies
# Source: Julia Quant Economoics 
function lorenz(v)  
    x = sort(v)
    S = cumsum(x)  # cumulative sums
    F = (1:length(v)) / length(v)
    L = S ./ S[end]
    return (; F, L) # returns named tuple
end

# Example of using the function 
w = exp.(randn(n));
gini(w)
(; F, L) = lorenz(w)
plot(F, L, label = "Lorenz curve, lognormal sample", legend = :topleft)
plot!(F, F, label = "Lorenz curve, equality")