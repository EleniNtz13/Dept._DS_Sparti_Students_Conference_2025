# The code creates the Steepest Descent Algorithm to find the minimum of a function of two variables
# The user enters the initial points, parameters and the function he wanted to examine
# The algorithm calculates the derivatives and the slope of the function, updating the points at each step. The process teminates:
# 1) when one of the termination criteria is satisfied (before the maximum number of iterations is exceeded), then the results are displayed on the user's screen
# 2) when the maximum number of iterations is exceeded, informing the user that the number of iterations has been exceeded without displaying the graphs

# The libraries used to execute the algorithm
import sympy as sp # Calculation of symbolic operations (e.g. for equations, for derivatives)
import numpy as np # Performing mathematical operations
import matplotlib.pyplot as plt # Used to create graphs and display results

# Using the Initial_Point() function to input the initial points (x0,y0) from the user
def Initial_Point(): # The Initial_Point() function
    while True: # If the user enters a non-numeric value, the user is prompted to enter the point again, informing the user with a corresponding message
        try: # Will be displayed on the screen if the user enters a numeric value
            print("Initial Point & Learning Rate: ") # The message is initially displayed on the user's screen
            x0 = float(input("x0 is the coordinate of the x'x axis. Please enter x0: ")) # Input of point x0 by the user
            y0 = float(input("y0 is the coordinate of the y'y axis. Please enter y0: ")) # User input of point y0
            return x0, y0 # Returns x0, y0

        except ValueError: # Will be displayed on the screen in case the user enters an incorrect input (a non-numeric value)
            print("Invalid input! Please enter valid numbers for x0 and y0.") # In this case, a corresponding message is displayed, informing the user and requesting the user to re-enter the point they typed incorrectly (using while True)

# Using the Parameters() function to input a (learning rate) and the constants c1, c2, c3 that terminate the algorithm
def Parameters(): # The Parameters() function
    while True: # If the user enters a non-numeric value, the corresponding parameter is requested again, informing the user with a corresponding message
        try: # Will be displayed on the screen in case the user enters correct input (numeric value)
            a = float(input("a is learning rate. Please enter a: ")) # Input of the learning rate (how fast the method progresses towards the minimum) by the user
            print("-------------------------------------------------------------------------------------")

            # The operation of the algorithm is described in the next command (print)
            print("The Steepest Descent algorithm stops if one of the following termination criteria is satisfied: \n"
            "1. In case the number of iterations exceeds 1000 and any of the criteria has not been satisfied.\n"
            "2. The slope of the function at the point found is less than a constant c1 defined by the user. \n"
            "If the slope is equal to 0 at the starting point (x0,y0), then it prompts the user for a new starting point.\n"
            "3. The distance between two consecutive points must be less than a constant c2 defined by the user. \n"
            "4. The value of the function between two consecutive points must be less than a constant c3 defined by the user. \n")
            print("-------------------------------------------------------------------------------------")

            # Input of constants and description of the operation of the corresponding termination criteria
            print("Therefore, for the algorithm to proceed, c1, c2, c3 are required from the user:")
            c1 = float(input("1st) What is c1 so that the criterion |∇f| ≤ c1 can be examined? ")) # The first criterion concerns the slope limit
            c2 = float(input("2nd) What is c2 so that the criterion |Xn - Xn-1| ≤ c2 can be examined? ")) # The second criterion concerns the limit for the distance between two consecutive points
            c3 = float(input("3rd) What is c3 so that the criterion |f(Xn) - f(Xn-1)| ≤ c3 can be examined? ")) # The third criterion concerns the limit for the difference between the values ​​of the function at successive points
            return a, c1, c2, c3 # Returns a, c1, c2, c3

        except ValueError: # Will be displayed on the screen in case the user enters an incorrect input (a non-numeric input)
            print("Invalid input! Please enter valid numbers.") # In this case, a corresponding message is displayed, informing the user and requesting the user to re-enter the parameter that was typed incorrectly (use of while True)

# Using the f() function so that the user can enter the desired function f(x,y) which will be minimized in symbolic form
# This returns the function in a form that can be processed algebraically by sympy
def f(): # The function f()
    while True: # If the user enters anything other than an equation, the function is requested again, informing the user with a corresponding message
        try: # Will be displayed on the screen when the user gives a valid function (e.g. 5*x + 4*y)
            print("-------------------------------------------------------------------------------------")
            expr_str = input("Please enter the function in x and y form (e.g., x**2 + y**4): ") # User input of the two-variable function
            expr = sp.sympify(expr_str) # Convert the string to an algebraic expression
            return expr # Returns the algebraic expression

        except sp.SympifyError: # Will be displayed on the screen if the user enters an invalid function (e.g. 5x + 4y)
            print("Invalid input! Please enter a valid function.") # In this case, a corresponding message is displayed, informing the user and requesting the input of a valid function again

# Implementation of the steepest_descent algorithm for minimizing a function of two variables
def steepest_descent(f_num, x0, y0, a, c1, c2, c3, derivative_x, derivative_y): # f_num: The function to minimize - converting the symbolic function to arithmetic using lambdify from sympy
    tries = 0 # tries: Represents the number of iterations of the algorithm, that is, how many times the algorithm will repeat the process for the function entered by the user
    MAX_TRIES = 1000 # MAX_TRIES: The maximum number of iterations. If the algorithm does not converge before reaching this number, then it will terminate
    points = [] # points: List that stores the points visited by the algorithm until it reaches the final point or MAX_TRIES
    x, y = sp.symbols('x y') # sp: From the sympy library
    # symbols: Function of the sympy library, used to create symbolic variables, with these variables not being numbers but symbols used in equations
    # x, y: symbols returns two symbolic variables which it assigns to x, y
    # So, it has two variables which can be used in derivation, solving equations, etc.

    x_path = [x0] # x_path: List that stores the value of x during the course of the algorithm, so that it can be visualized
    y_path = [y0] # y_path: List that stores the value of y during the course of the algorithm, so that it can be visualized
    z_path = [f_num(x0, y0)] # z_path: List that stores the value of f(x, y) during the course of the algorithm, so that it can be visualized

    slope_x = derivative_x.subs({x: x0, y: y0}).evalf() # Calculate the derivative value at the point (x0, y0), convert the result to a numerical value and store it in the slope_x variable
    # derivative_x: The partial derivative of the function with respect to the variable x
    # .subs({x: x0, y: y0}): Method used to replace variables x, y with values ​​x0, y0 respectively
    # .evalf(): Method that converts symbolic expression to numeric value and returns it

    slope_y = derivative_y.subs({x: x0, y: y0}).evalf() # In the same way here

    grad_norm = sp.sqrt(slope_x**2 + slope_y**2) # Calculates the measure of the scalar which shows how fast the function changes at the point we are examining
    # grad_norm: Calculates the measure (norm) of the gradient vector (gradient norm)
    # If it is zero, it means that the function is already at a local minimum or maximum
    # sp.sqrt: Calculate the square root of the sum of the squares of the derivatives

    slope_x_expr = derivative_x # The value of the partial derivative with respect to x is assigned to the variable slope_x_expr
    slope_y_expr = derivative_y # The value of the partial derivative with respect to y is assigned to the variable slope_y_expr
    slope_x = slope_x_expr.subs({x: x0, y: y0}).evalf() # Replace x,y by x0,y0 & calculate the numerical value of the partial derivative with respect to x at the point (x0, y0)
    slope_y = slope_y_expr.subs({x: x0, y: y0}).evalf() # Replace x,y by x0,y0 & calculate the numerical value of the partial derivative with respect to y at the point (x0, y0)
    print("Η μερική παράγωγος ως προς x:", slope_x_expr) # Display the partial derivative with respect to x
    print("Η μερική παράγωγος ως προς y:", slope_y_expr) # # Display the partial derivative with respect to y

    # In case the slope is 0, we ask the user to give new x0 and y0
    while grad_norm == 0:
        print("The slope at point (x0, y0) is 0. Please enter a new starting point.")
        x0, y0 = Initial_Point() # The user re-enters the new points

        # We recalculate the derivatives and the slope at the new point
        slope_x = derivative_x.subs({x: x0, y: y0}).evalf()
        slope_y = derivative_y.subs({x: x0, y: y0}).evalf()
        grad_norm = sp.sqrt(slope_x**2 + slope_y**2)

    # As long as the algorithm repeats the process and terminates after 1000 iterations
    while tries <= MAX_TRIES:
        # Calculating derivatives and gradients
        slope_x = derivative_x.subs({x: x0, y: y0}).evalf()
        slope_y = derivative_y.subs({x: x0, y: y0}).evalf()
        grad_norm = sp.sqrt(slope_x**2 + slope_y**2)

        # Add the point with coordinates (x0,y0) to the points list using the append method
        points.append((x0, y0))

        # TERMINATION CRITERIA
        # Criterion 1 -> Check if the slope is less than the constant c1:
        # If it is, the process is stopped and the result is recorded in criterion
        # If not, check if there are at least two points in the points list. If there are, calculate the distance of the last and penultimate point, based on the Euclidean distance formula
        if grad_norm < c1: # grad_norm: the value of the gradient. If the value of the gradient is less than the constant c1 defined above, then the algorithm proceeds to the next instructions after the 1st criterion is satisfied
            criterion = "1st criterion: The slope is small." # If the above condition is true, then the variable criterion is assigned the value "1st criterion: The slope is small.", which recognizes that the 1st criterion has been satisfied
            break # If the above is true, then the loop execution stops and the process is completed

        elif len(points) > 1: # If the first criterion is not satisfied, the case is considered where the slope value is greater than the constant c1
            prev_x, prev_y = points[-2] # points[-2]: Refers to the second-to-last element of the points list
            # Therefore, the command "prev_x, prev_y = points[-2]" assigns the coordinates of the penultimate point in the list to the variables prev_x, prev_y
            distance = sp.sqrt((x0 - prev_x)**2 + (y0 - prev_y)**2).evalf() # (x0 - prev_x)**2 + (y0 - prev_y)**2: Calculate the square of the distance between two points in two-dimensional space
            # .evalf(): Calculates the numerical value of the result, converting the symbolic expression to a decimal number and storing it in the distance variable

            # Criterion 2 -> Check if the distance between the current and the penultimate point is less than the constant c2:
            # If it is, the process is stopped and the result is recorded in criterion
            # If it is not, the values ​​of the function at the penultimate and current points are calculated and stored in the variables f_prev and f_current respectively, in order to complete the optimization process and find the minimum point
            if distance < c2: # distance: The distance shows us how far apart the two points are in two-dimensional space (it has been calculated above)
            # When the distance between two consecutive points is less than the value of the constant c2 entered by the user, then the condition is true and the points are very close to each other
                criterion = "2nd criterion: The distance between two consecutive points is small." # The value "2nd criterion: The distance between two consecutive points is small." is assigned to the variable criterion
                break  # If the above is true, then the loop execution stops and the process is completed

            f_prev = f_num(prev_x, prev_y) # Calculate the value of the function f_num at the penultimate point (prev_x, prev_y), which is the previous point from (x0, y0)
                                           # The value of the function at the penultimate point is stored in the variable f_prev
            f_current = f_num(x0, y0) # Calculate the value of the function f_num at the current point (x0, y0)
                                      # The value of the function at the current point is stored in the variable f_current

            # Criterion 3 -> Convergence of the difference of values ​​of the function. Checks if the absolute difference between the values ​​of a function at two consecutive points is less than the constant c3:
            # If it is, then the function has converged and the convergence criterion is recorded
            if abs(f_current - f_prev) < c3: # abs(): Absolute value function
            # Here, the absolute difference between the values ​​of the function at two consecutive points is calculated
            # If this difference is less than the constant c3, then it does not change much from one point to another
            # This means that the optimization process is approaching the desired result

                criterion = "3rd criterion: The convergence of the function is small." # If the above condition is true, then the value "3rd criterion: The convergence of the function is small." is assigned to the variable criterion
                break # If the above is true, then the loop execution stops and the process is completed. The process is aborted because there are no significant changes in the function values
            # This means that the process is complete since the function values ​​are no longer changing significantly
            # Thus, the function is approaching its minimum point

        # Update variables x0 and y0. The values ​​are updated based on the slope and learning rate (as demonstrated in the following operations)
        x0 = x0 - a * slope_x
        y0 = y0 - a * slope_y

        x_path.append(x0) # Append the current value of x, i.e. x0 each time, to the x_path list, so that this list contains all the points x visited by the algorithm throughout the process
                          # Thus, the list displays the points that x has passed as it approaches the optimum
        y_path.append(y0) # Append the current value of y, i.e. y0 each time, to the y_path list, so that this list contains all the y points visited by the algorithm throughout the process
                          # # Thus, the list displays the points that y has passed as it approaches the optimum
        z_path.append(f_num(x0, y0)) # Append the function values ​​at each (current) point (x0, y0) to the z_path list
                                     # This way, we know how the function value changes as we approach the optimum during the algorithm iterations
        
        # Go to the next cycle of repeating the process to check the next points
        tries += 1

    # If the algorithm fails to converge to an optimal point within the specified number of iterations, then there is a failure
    if tries > MAX_TRIES: # If the process for finding the minimum exceeds 1000 iterations...
        criterion = f"Η αναζήτηση απέτυχε: μέγιστες επαναλήψεις ({MAX_TRIES})." # ...informs the user about the reason the algorithm failed to find the optimum
        return x0, y0, None, criterion, False, x_path, y_path, z_path, tries # Return the specified points, parameters, criterion and number of iterations

    # If the execution is successful, then the following is returned:
    # # The final values ​​of the parameters, the final value of the function, a success message & success status (True), the path of the values ​​x, y, f(x,y) and the total number of iterations
    return x0, y0, f_num(x0, y0), criterion, True, x_path, y_path, z_path, tries

# Use the main function main(), which executes the entire program. It is responsible for executing the main program
def main(): #The function main()
    x0, y0 = Initial_Point() # Asks the user for the initial points (x0, y0), calling the Initial_Point() function
    a, c1, c2, c3 = Parameters() # Asks the user for the learning rate and criterion termination constants (a, c1, c2, c3) by calling the Parameters() function

    f_sym = f() # Calls the function f(), which defines the symbolic function we want to optimize
    x, y = sp.symbols('x y') # Creates the symbolic variables x, y with the Sympy library

    # Convert the function from symbolic to numerical function that can calculate the values ​​of x, y
    f_num = sp.lambdify((x, y), f_sym, "numpy")

    derivative_x_sym = sp.diff(f_sym, x) # Calculate the partial derivative of the function f(x,y) with respect to x using Sympy
    derivative_y_sym = sp.diff(f_sym, y) # # Calculate the partial derivative of the function f(x,y) with respect to y using Sympy 

    # Save results after running the code
    min_x, min_y, min_value, criterion, show_plots, x_path, y_path, z_path, total_tries = steepest_descent(f_num, x0, y0, a, c1, c2, c3, derivative_x_sym, derivative_y_sym)

    # If the maximum number of iterations is exceeded, then the graphs are not displayed
    if not show_plots:
        print("Maximum number of repetitions exceeded.") # This specific message is displayed informing the user
        return

    # The algorithm will read the following commands, in case I have:
    print("-------------------------------------------------------------------------------------")
    print(f"Minimum point: ({min_x}, {min_y}), with function value f(x, y) = {min_value}") # 1) The values ​​of the minimum point, 2) the value of the function
    print("Convergence criterion ->", criterion) # 3) The satisfied criterion
    print(f"Αριθμός επαναλήψεων: {total_tries}") # 4) The number of repetitions

    # If the above results are valid, then 3D and 2D graphs are created which illustrate the progress of the algorithm
    x_vals = np.linspace(-1.5, 1.5, 400) # It concerns the 400 intermediate values ​​from -1.5 to 1.5 for the x-axis
    y_vals = np.linspace(-1.5, 1.5, 400) # It concerns the 400 intermediate values ​​from -1.5 to 1.5 for the y-axis
    X, Y = np.meshgrid(x_vals, y_vals) # Create X (contains all x-axis coordinates) and Y (contains all y-axis coordinates) arrays
    Z = f_num(X, Y) # Calculate the value of the function f_num for each pair (X,Y). So, Z represents the height of the graph
    fig = plt.figure(figsize=(14, 6)) # Create a shape with dimensions: 14 inches wide and 6 inches high
    # figsize: Window size so the graph is large

    # 3D Plot (3 dimensions: X, Y, Z)
    ax1 = fig.add_subplot(121, projection='3d')
    # Create a subgraph in the first space with a 1x2 layout (one row and two columns), which has a three-dimensional form (projection='3d')

    ax1.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis') # Create a surface
    # alpha=0.3: Create a translucent surface so that the elements underneath can be seen
    # cmap='viridis': This concerns the colors that will depict the surface ("viridis" style)

    ax1.plot(x_path, y_path, z_path, 'r-', label='Steepest Descent Path') # This function plots the path of the Steepest Descent algorithm
    # x_path, y_path, z_path: Lists containing the coordinates of the algorithm path
    # r-: Sets the color and type of the line (r: red, -: solid)
    # label='Steepest Descent Path': Sets the label

    ax1.scatter(min_x, min_y, min_value, color='red', s=100, label='Ελάχιστο Σημείο') # Placing the minimum
    # min_x, min_y, min_value: Contains the coordinates of the minimum
    # color='red': Sets the color of the minimum
    # s=100: Size of the minimum (dot)
    # label='Minimum Point': Sets the label

    # Defines the labels for the x, y, z axes
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$'),
    ax1.set_zlabel('$f(x, y)$')

    # Add a caption to display the labels we have defined for each element
    ax1.legend()


    # 2D Contour Plot (Contour Curves)
    ax2 = fig.add_subplot(122)
    # Create a new subgraph in the second space with a 1x2 layout (one row and two columns)

    ax2.contour(X, Y, Z, levels=50, cmap='viridis') # Draw contour curves
    # levels=50: Defines the number of contour lines to draw
    # cmap='viridis': Refers to the colors that will depict the curves ("viridis" style)

    ax2.plot(x_path, y_path, 'r-', label='Steepest Descent Path') # This function draws the path of the Steepest Descent algorithm
    # x_path, y_path: Lists containing the coordinates of the path of the algorithm
    # r-: Sets the color and type of the line (r: red, -: solid)
    # label='Steepest Descent Path': Sets the label

    ax2.scatter(min_x, min_y, color='red', s=100, label='Ελάχιστο Σημείο') # Placing the minimum
    # min_x, min_y: Contains the coordinates of the minimum
    # color='red': Sets the color of the minimum
    # s=100: Size of the minimum (dot)
    # label='Minimum Point': Sets the label

    # Defines the labels for the x, y axes
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')

    # Add a caption to display the labels we have defined for each element
    ax2.legend()

    # Display 3D and 2D graphs
    plt.show()

# Completing main

main()






