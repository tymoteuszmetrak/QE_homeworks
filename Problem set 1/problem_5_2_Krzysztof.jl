using DelimitedFiles, Plots, Statistics

function count_positive_numbers(arr)
    sum(arr .> 0)
end

count_positive_numbers([1, 0 , -1])
count_positive_numbers([-4, 10, 14, -2])