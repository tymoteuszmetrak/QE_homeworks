using DelimitedFiles, Plots, Statistics #Hincz, Kuba, Mętrak, Szymczak

# Problem 1: Odd or Even
# a) with validation of integers
function odd_or_even(n)
    if typeof(n) == Int64
        if iseven(n) == true
            println("Even")
        else
            println("Odd")
        end
    else 
        println("not an integer")
    end
end

# Examples 
odd_or_even(9)
odd_or_even(6)
odd_or_even(0)
odd_or_even(-3)
odd_or_even(-4)
odd_or_even(4.5)

# b - without validation of integers
function odd_or_even_2(x)
    if iseven(x) == true
        println("Even")
    else
        println("Odd")
    end
end

# Examples 
odd_or_even_2(9)
odd_or_even_2(6)
odd_or_even_2(0)
odd_or_even_2(-3)
odd_or_even_2(-4)
odd_or_even_2(4.5) #incorrect

# Problem 2: Boolean operators
function compare_three(a, b, c)
    if (a > 0 && b > 0 && c > 0)
        println("All numbers are positive")
    elseif (a == 0 && b == 0 && c == 0) #2nd condition, because it is a special case of "At least one number is not positive"
        println("All numbers are zero")
    else
        println("At least one number is not positive")
    end
end

compare_three(1, 2, 3) # Output: All numbers are positive
compare_three(-1, 5, 7) # Output: At least one number is not positive
compare_three(0, -4, 3) # Output: At least one number is not positive
compare_three(0, 0, 0) # Output: All numbers are zero


# Problem 3: Factorial Calculation Using a Loop
function my_factorial(n)
    result = 1
    for i in 1:n
        result = result * i
    end
    return result 
end

my_factorial(0) #1
my_factorial(1) #1
my_factorial(2) #2
my_factorial(5) # Output: 120 (because 5! = 1 × 2 × 3 × 4 × 5 = 120)
my_factorial(7) # Output: 5040 (because 7! = 1 × 2 × 3 × 4 × 5 × 6 × 7 = 5040)


# Problem 4: Count Positive Numbers Using a Loop
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


# Problem 5: Plotting Powers of x Using a Loop
function plot_powers(n)
    if n>0 && isinteger(n)     
        power_plot = plot()
        for i in 1:n
            powers(x) = x^i
            plot!(powers, -10:0.2:10, label = "x^$i", linestyle =:dash, linewidth = 3)
        end
        xaxis!("x")
        yaxis!("y")
        title!("Powers of x")
        return power_plot
    else
        println("You should have provided a POSITIVE INTEGER")
    end
end

plot_powers(-3)
plot_powers(2.34)
plot_powers(2)
my_plot = plot_powers(3)
my_plot


# Problem 5 prim: Count Positive Numbers Using Broadcasting
function count_positives_broadcasting(arr)
    return sum(arr .> 0)
end
count_positives_broadcasting([1, -3, 4, 7, -2, 0]) # 3
count_positives_broadcasting([-5, -10, 0, 6]) # 1


# Problem 6: Standard Deviation Using Broadcasting and Vectorized Operations
function standard_deviation(x)
    μ = sum(x)/length(x)
    d = x .- μ
    squared_d = d.^2
    variance = sum(squared_d)/(length(x)-1)
    SD = variance^(0.5)
    return SD
end

standard_deviation([1,1,1,1]) #0.0 as expected
standard_deviation([1, 2, 3, 4, 5]) # Output: 1.5811388300841898
standard_deviation([5, 10, 15]) # Output: 5.0
standard_deviation(2:7) # Output: 1.8708286933869707


# Problem 7: Import the Data, Plot It, and Calculate Correlations
pwd()

data = readdlm("Problem set 1//dataset.csv", ',',Int64)
# each column contains the data for earnings, education, and hours worked, respectively 

plot_1 = scatter(data[:,2],data[:,1], #education (2nd column) as x, earnings (1st column) as y
label = "Earnings vs. Education",
xaxis="Education",
yaxis="Earnings",
title="Relationship between education and earnings",
markercolor = :green)

plot_2 = scatter(data[:,3], data[:,1],
label = "Earnings vs. Hours worked",
xaxis="Hours worked",
yaxis="Earnings",
title="Relationship between hours worked and earnings",
markercolor=:red)

# correlation between earnings and education (first number), and hours worked (second)
cor(data)[1,2:3] #0.39; 0.25

# Discussion

# There is a small positive correlation between hours worked and earnings at ca. 0.25.
# There seems to be a minute average increase of earnings with hours worked until the hours worked reach 70, then the earnings seem to slightly decrease. 
# Numerically, the correlation is low also because e.g. people earning slightly below 4.0x10^5 
# seem to work various hours which are distributed roughly uniformly
# Additionally, we might suspect a feedback loop between x = hours worked and y = earnings, 
# which would result in an inconsistent estimate of the effect of x on y in a hypothetical simple regression. 
# Ceteris paribus, people who work more should earn more as they exhibit more effort, 
# but on the other hand leisure is a normal good for many people who would like to work less if they earned more.


# The correlation between education and earnings is stronger and more straightforward
# though still relatively mild at around 0.39.
# It may be concluded that the more years of formal education, the more person earns.
#This is probably because additional education allows to gain incremental knowledge and skills 
#valuable to some extent in the labour market. 
#Moreover, additional passed years of education signal intrinsic ability.

# All in all, this is probably not the best method to calculate the correlation
#because education is a discrete variable.
# It may make more sense to compare means of earnings between groups of people with given years of education.
#it is important to bear in mind that correlation is a linear measure of association, but both examined associations could be of non-linear variety.  

#for both of the above relations 
#more formal analysis (e.g. multivariate regression) is advised in order to draw more reliable conclusions
