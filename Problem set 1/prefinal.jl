using DelimitedFiles, Plots, Statistics #Hincz, Kuba, Mętrak, Szymczak

# Problem 1 
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

# Problem 2 
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
