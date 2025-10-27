import numpy as np

def check_timesteps_are_equal(sim_data, keys, col_name):
    """
    Extracts the value of a specific column (col_name) from the first row 
    of the DataFrames specified by keys, and checks if all values are equal.
    Raises a ValueError if they are not.
    """    
    # Extract all relevant timestep values
    timesteps = {}
    for key in keys:
        if key in sim_data:
            # Assuming the value is in the first row (index 0) of the DataFrame
            value = sim_data[key].loc[0, col_name]
            timesteps[key] = value
        else:
            raise KeyError(f"DataFrame key '{key}' not found in sim_data dictionary.")

    # Check for equality using a set
    # The length of the set of values must be 1 for all values to be identical
    unique_timesteps = set(timesteps.values())
    
    if len(unique_timesteps) == 1:
        print(f"✅ Success: All specified DataFrames have the same '{col_name}' value: {list(unique_timesteps)[0]}")
    else:
        # Halt the program and return an error message if not equal
        error_details = ", ".join([f"{k}: {v}" for k, v in timesteps.items()])
        error_message = (
            f"ERROR: The '{col_name}' values are NOT uniform across all DataFrames. "
            f"Found values: {error_details}"
        )
        raise ValueError(error_message)

def print_terminal_table(data, title=None):
    """
    Prints a table to the terminal with an optional title.
    """
    # 1. Determine the maximum width for each column
    column_widths = []
    num_cols = len(data[0])
    
    for col in range(num_cols):
        max_width = 0
        for row in range(len(data)):
            # Ensure all data elements are strings before checking length
            max_width = max(max_width, len(str(data[row][col])))
        # Add a small padding (e.g., 2 spaces)
        column_widths.append(max_width + 2)

    # Calculate the total width of the inner table content (excluding borders)
    total_inner_width = sum(column_widths) + (num_cols - 1)
    
    # Calculate the total width of the table including borders
    total_table_width = total_inner_width + 2

    # Helper function to print a separator line
    def print_separator():
        line = "+"
        for width in column_widths:
            line += "-" * width + "+"
        print(line)

    # 2. Print the Title Row (if provided)
    if title:
        # Create a title separator that spans the whole width
        title_separator = "=" * total_table_width
        print(title_separator)
        
        # Center the title within the table width (excluding the outer borders)
        # Use str.center() to center the title string
        title_row = f"|{title.center(total_inner_width)}|"
        print(title_row)
        print(title_separator)
    
    # 3. Print the data rows (Header and Values)
    print_separator()
    for row in data:
        row_string = "|"
        for i, cell in enumerate(row):
            # Convert cell to string and left-align
            cell_str = str(cell)
            row_string += cell_str.ljust(column_widths[i]) + "|"
        print(row_string)
        print_separator()

def calculate_ccv(g, a, b, dt):
    """
    Calculates the cumulative (integrated) constraint violation for the 
    inequality constraint: a(t) < g(t) < b(t).

    The cumulative violation is defined as the sum (or integral) of the 
    magnitude of the constraint violation over the entire trajectory.

    Args:
        g (np.ndarray): The trajectory (the signal/state being constrained).
        a (np.ndarray): The lower constraint trajectory.
        b (np.ndarray): The upper constraint trajectory.
        dt (float): The time step duration (assuming a constant time step).

    Returns:
        float: The cumulative constraint violation.
    """
    # Calculate the Violation for the Lower Bound (g < a)
    # The violation is: max(0, a - g)
    # If g is too low (g < a), the violation is (a - g), otherwise 0.
    lower_violation = np.maximum(0, a - g)    
    # Calculate the Violation for the Upper Bound (g > b)
    # The violation is: max(0, g - b)
    # If g is too high (g > b), the violation is (g - b), otherwise 0.
    upper_violation = np.maximum(0, g - b)    
    # Sum the Violations at Each Time Step
    # The total pointwise violation nu(t) is the sum of the upper and lower violations.
    total_pointwise_violation = lower_violation + upper_violation    
    # Calculate the Cumulative Violation (Integration)
    # This is an approximation of the integral: sum(nu(t) * dt)
    cumulative_violation = np.sum(total_pointwise_violation) * dt    
    return cumulative_violation

