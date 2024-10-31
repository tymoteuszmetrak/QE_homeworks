using Plots, DelimitedFiles, Statistics

function standard_deviation(x)
    n = length(x)
    μ = sum(x) / n
    d = x .- μ
    σ = sqrt(sum(d.^2) / (n-1))
end

standard_deviation([1, 2, 3, 4])