# Problem Set 1 - Quantitative Economics 2024
# Wojciech Szymczak
cd("C:/Users/wojci/OneDrive/WSZ/Quantitative Economics/PS1")
pwd()
import Pkg
Pkg.add("DelimitedFiles")
Pkg.add("Plots")
Pkg.add("Statistics")


# Exercise 1 
function odd_or_even(x)
    if iseven(x) == true
        print("Even")
    else
        print("Odd")
    end
end

# Examples 
odd_or_even(9)
odd_or_even(6)
odd_or_even(0)

# Exercise 2 
function compare_three(x, y, z)
    if (x > 0 && y > 0 && z > 0)
        print("All numbers are positive")
    elseif (x <= 0 || y<= 0 || z<= 0)
        print("At least one number is not positive")
    elseif (x == 0 && y == 0 && z == 0)
        print("All numbers are equal to zero")
    end
end

compare_three(1, 2, 3) # Output: All numbers are positive
compare_three(-1, 5, 7) # Output: At least one number is not positive
compare_three(0, -4, 3) # Output: At least one number is not positive
compare_three(0, 0, 0) # Output: All numbers are zero

# Problem 3 Factorial loop
function my_factorial(x)
    result = 1
    for i = 1:x
        result = result * i
    end
    return result 
end

my_factorial(5) # Output: 120 (because 5! = 1 × 2 × 3 × 4 × 5 = 120)
my_factorial(7) # Output: 5040 (because 7! = 1 × 2 × 3 × 4 × 5 × 6 × 7 = 5040)

# Problem 4 count positive number in an array
function count_positives(arr)
    counter = 0
    for i in arr
        if (i>0)
            counter = counter + 1
        end
    end
    return counter
end

count_positives([1, -3, 4, 7, -2, 0]) # Output: 3
count_positives([-5, -10, 0, 6]) # Output: 1

# Problem 5 Plotting powers of X using loop 
using Plots
function plot_powers(n)
    plot(xlims = (-10, 10),
    title = "Powers of x",
    xlabel = "x",
    ylabel = "y")
    x = Array(-10:0.2:10);
    p = 1 
    while p - 1 < n
        plot!(x, x.^p,
        linestyle =:dash,
        linewidth = 3,
        label = "x^$p")
        p = p + 1
    end
    display(current())
end

# Example
plot_powers(3)

# Problem 6 Count positive using broadcasting
function count_positives_broadcasting(arr)
    b = [0]
    return sum(broadcast(>, arr, b))
end

count_positives_broadcasting([1, -3, 4, 7, -2, 0]) # 3
count_positives_broadcasting([-5, -10, 0, 6]) # 1

# Problem 7 Standard Deviation
function standard_deviation(arr)
    mu = sum(arr) / length(arr)
    d = broadcast(-, arr, mu)
    squared_d = d.^2
    return sqrt( sum(squared_d) / (length(arr) - 1) )
end

standard_deviation([1, 2, 3, 4, 5]) # Output: 1.5811388300841898
standard_deviation([5, 10, 15]) # Output: 5.0
standard_deviation(2:7) # Output: 1.8708286933869707

# Problem 8 Plot graph and statistics
using Statistics, Plots, DelimitedFiles
data = readdlm("data\\dataset.csv", ',', Float64)

# Plot Education vs Earnings
plot()
scatter(data[:, 2], data[:,1], 
title = "Relationship between education and earnings",
color = "green",
xlabel = "Education",
ylabel = "Earnings")

# Plot Hours worked vs Earnings
plot()
scatter(data[:, 3], data[:,1],
title = "Relationship between hours worked and earnings",
color = "red",
xlabel = "Hours worked",
ylabel = "Earnings")

# Calculate correlations
cor(data[:, 2], data[:,1]) # Education
cor(data[:, 3], data[:,1]) # Hours worked

# Findings

# There is a positive correlation between education and earnings.
# However, this is not the best method to calculate the correlation
# Education is a discrete variable 
# It may make more sense to compare earnings by education groups 

# There is a positive correlation between hours worked and earnings
# However, the correlation coefficient is small (<0.3)
