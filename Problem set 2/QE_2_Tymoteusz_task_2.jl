# Problem 2

using LinearAlgebra
using PrettyTables

# At first we calculate an exact solution
function exact(α, β)
    x = [1 1 1 1 1]
    return(x)
end

# The α and β may take any value, they are not needed for calculations
exact(0,0)


# Then we proceed to a function, that present both exact solution 
# and the one obtained with backslash operator
function solver(α, β)
    # The eexact solution
    x_exact = [1 1 1 1 1]
    
    # We define matrix A and matrix b
    A = [1 0 0 0 0; -1 1 0 0 0; 0 -1 1 0 0; (α-β) 0 -1 1 0; β 0 0 -1 1]'
    b = [α, 0, 0, 0, 1]

    # Calculating the vector x with backslash operator
    x_backslash = A\b

    # Relative residual and condition number
    relative_residual = norm(A*x_backslash - b)
    condition_number = cond(A)

    return(x_exact, x_backslash, relative_residual, condition_number)
end

# Let's look at selected outcomes depending on β
result = solver(0.1, 1)
result = solver(0.1, 10)
result = solver(0.1, 100)
result = solver(0.1, 1000)
result = solver(0.1, 1000000000000)


# In order to create a table that shows x1  the condition number, and the relative residual
# we build  a function that do so for a given number of β.
# We supply a maximum power, to which we take 10 and calculate β.


function create_table(power)
    # Powers of 10
    n = []
    # Calculated β
    betas = []
    # Exact values of x_1 always equal 1
    x_1_exact = []
    # x_1 values calculated using backslash operator
    x_1_values = []
    # Condition number
    cond_num_values = []
    # relative residual
    rel_resid_values = []

    for i in 0:power
        n = push!(n, i)
        betas = push!(betas, 10^i)
        x_1_exact = push!(x_1_exact, 1)
        result_now = solver(0.1, 10^i)
        x_1_values = push!(x_1_values, result_now[2][1])
        cond_num_values = push!(cond_num_values, result_now[4])
        rel_resid_values = push!(rel_resid_values, result_now[3])
    end
    data = hcat(n, betas, x_1_exact,x_1_values, cond_num_values, rel_resid_values)
    header = (["Power of 10", "β", "Exact x_1", "x_1 using backslash operator", "Condition number", "Relative residual"])
    pretty_table(data;
        header = header,
        header_crayon = crayon"yellow bold",
        tf = tf_unicode_rounded
        )
end

# Now, we create a table!
create_table(12)
