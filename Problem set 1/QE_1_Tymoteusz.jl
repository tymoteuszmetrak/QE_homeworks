# Problem 1
function odd_or_even(n)
    isodd(n) ? message="Odd" : message="Even"
    return  message
end

odd_or_even(6)
odd_or_even(7)

# We can also check whether the imput is number
function odd_or_even2(n)
    if n isa Number
        odd_or_even(n)
    else
        message = "Given value should be a number"
        return message 
    end
    
end

odd_or_even2(231)

# Problem 2 
function compare_three(a,b,c)
    if a>0 && b>0 && c>0
        message = "All the numbers are positive"
    elseif a==0 && b==0 && c==0
        message = "All numbers are zero"
    else
        message = "At least one number is not positive"
    end
    return message
end

compare_three(12,6,5)
compare_three(12,-6,5)
compare_three(0,0,0)

# Problem 3
function my_factorial(n)
    wektor = collect(1:n)
    outcome = 1
    for i in wektor
        outcome = outcome*i
    end
    return outcome
end

my_factorial(5)

# Problem 4
function count_positives(array)
    outcome = 0
    for i in array
        if i>0
            outcome += 1
        end
    end
    return outcome
end

count_positives([1,2,-5])


# Problem 5
using Plots

function plot_powers(n)
    if n <=0 
        println("You should have provided a POSITIVE integer!")
    elseif !isinteger(n)
        println("You should have provided an INTEGER")
    else
        #y(x)=x
        #plot(y, -10:0.1:10)
        plt = plot()
        for i in collect(1:n)
            powers(x) = x^i
            plot!(powers, -10:0.1:10)
        end
        display(plt)
    end
end

plot_powers(-3)
plot_powers(2.34)
plot_powers(3)

# Problem 5 prim

function count_positives_broadcasting(array)
    comp = zeros(length(array))
    z = array.>comp
    return sum(z)
end

count_positives_broadcasting([0,1,3,-12])

# Problem 6
function standard_deviation(x)
    mean = sum(x)/length(x)
    d = x .- mean
    squared_d = d.^2
    variance = sum(squared_d)/(length(x)-1)
    sd = variance .^(0.5)
    return sd
end

standard_deviation([1,1,1,1])
standard_deviation([1,2,3,4,5])
standard_deviation([5,10,15])
standard_deviation(2:7)

# Problem 7
using DelimitedFiles, Plots, Statistics
pwd()

data = readdlm("problem_sets//PS1//dataset.csv", ',',Int64)
# each column contains the data for earnings, education, and hours worked, respectively 

plot_1 = (scatter(data[:,2],data[:,1],
label = "Earnings vs. Education",
xaxis="Education",
yaxis="Earnings",
title="Relationship between education and earnings",
markercolor = :green)
)

plot_2 = (scatter(data[:,3], data[:,1],
label = "Earnings vs. Hours worked",
xaxis="Hours worked",
yaxis="Earnings",
title="Relationship between hours worked and earnings",
markercolor=:red)
)

# correlation between earnings and education (first number), and hours worked (second)
cor(data)[1,2:3]

# Discussion
# There is only a mild correlation between hours worked and earnings.
# Apparentl=y, it is due to a considerable number of outstanding observations - people earning 4.0x10^5 and more regardless the number of hours worked.
# Additionally, the relationship is not straightforward.
# Up to around 70  hours per week earnings seem to  increase on average, but then they seem to decrease.


# More certain relationship is between education and earnings.
# The correlation is stronger (though still relatively mild - around 0.39).
# It may be concluded that the more years of formal education, the more person earns.
# However, there are still many outstanding observations.

# More formal analysis (e.g. regression) is adviced in order to draw more reliable conclusions.
