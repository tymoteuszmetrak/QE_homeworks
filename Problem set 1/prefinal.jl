using DelimitedFiles, Plots, Statistics #Hincz, Kuba, Mętrak, Szymczak

# Problem 1: Odd or Even
#a) with validation of integers
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

#b - without validation of integers
function odd_or_even_2(x)
    if iseven(x) == true
        println("Even")
    else
        println("Odd")
    end
end

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

