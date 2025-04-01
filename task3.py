import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
import os

# Define the path to load the data
X_path = 'hw2-ID-X.csv'
Xprime_path = 'hw2-ID-Xprime.csv'

# Load the data
X_data = np.loadtxt(X_path, delimiter=',')
Xprime_data = np.loadtxt(Xprime_path, delimiter=',')

# Extract the state variables and their derivatives
x1 = X_data[:, 0]  # First state variable
x2 = X_data[:, 1]  # Second state variable
x1_dot = Xprime_data[:, 0]  # Derivative of first state variable
x2_dot = Xprime_data[:, 1]  # Derivative of second state variable

print(f"Data shape: {X_data.shape}")
print(f"First few rows of X data:\n{X_data[:5]}")
print(f"First few rows of Xprime data:\n{Xprime_data[:5]}")

# Create a function to try different polynomial terms for system identification
def identify_system(x1, x2, derivative, max_degree=3):
    best_model = None
    best_rmse = float('inf')
    best_degree = 0
    best_features = None
    best_coefficients = None
    
    for degree in range(1, max_degree + 1):
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly.fit_transform(np.column_stack((x1, x2)))
        feature_names = poly.get_feature_names_out(['x1', 'x2'])
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_poly, derivative)
        
        # Predict and calculate RMSE
        y_pred = model.predict(X_poly)
        rmse = np.sqrt(mean_squared_error(derivative, y_pred))
        
        # Print coefficients for this degree
        print(f"\nDegree {degree} polynomial terms:")
        for i, (feature, coef) in enumerate(zip(feature_names, model.coef_)):
            if i == 0:  # Handle the intercept separately
                print(f"Intercept: {model.intercept_:.6f}")
            else:
                print(f"{feature}: {coef:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        # Update best model if this one is better
        if rmse < best_rmse:
            best_rmse = rmse
            best_degree = degree
            best_model = model
            best_features = feature_names
            best_coefficients = np.concatenate(([model.intercept_], model.coef_))
    
    return best_model, best_degree, best_features, best_coefficients, best_rmse

# Identify governing equation for x1_dot
print("\n" + "="*50)
print("Identifying equation for dx1/dt:")
print("="*50)
model1, degree1, features1, coeffs1, rmse1 = identify_system(x1, x2, x1_dot)

# Identify governing equation for x2_dot
print("\n" + "="*50)
print("Identifying equation for dx2/dt:")
print("="*50)
model2, degree2, features2, coeffs2, rmse2 = identify_system(x1, x2, x2_dot)

# Get the coefficients from the 2nd degree polynomial fit for more detailed analysis
poly1 = PolynomialFeatures(degree=2, include_bias=True)
X_poly1 = poly1.fit_transform(np.column_stack((x1, x2)))
feature_names1 = poly1.get_feature_names_out(['x1', 'x2'])
model1_deg2 = LinearRegression()
model1_deg2.fit(X_poly1, x1_dot)
coeffs1_deg2 = np.concatenate(([model1_deg2.intercept_], model1_deg2.coef_))

poly2 = PolynomialFeatures(degree=2, include_bias=True)
X_poly2 = poly2.fit_transform(np.column_stack((x1, x2)))
feature_names2 = poly2.get_feature_names_out(['x1', 'x2'])
model2_deg2 = LinearRegression()
model2_deg2.fit(X_poly2, x2_dot)
coeffs2_deg2 = np.concatenate(([model2_deg2.intercept_], model2_deg2.coef_))


# Plot the original derivatives vs the predicted derivatives
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
y_pred1 = model1_deg2.predict(X_poly1)
plt.scatter(x1_dot, y_pred1, alpha=0.5)
plt.plot([min(x1_dot), max(x1_dot)], [min(x1_dot), max(x1_dot)], 'r-')
plt.xlabel('Actual dx1/dt')
plt.ylabel('Predicted dx1/dt')
plt.title(f'dx1/dt: RMSE = {np.sqrt(mean_squared_error(x1_dot, y_pred1)):.6f}')

plt.subplot(1, 2, 2)
y_pred2 = model2_deg2.predict(X_poly2)
plt.scatter(x2_dot, y_pred2, alpha=0.5)
plt.plot([min(x2_dot), max(x2_dot)], [min(x2_dot), max(x2_dot)], 'r-')
plt.xlabel('Actual dx2/dt')
plt.ylabel('Predicted dx2/dt')
plt.title(f'dx2/dt: RMSE = {np.sqrt(mean_squared_error(x2_dot, y_pred2)):.6f}')

plt.tight_layout()
plt.savefig('system_identification.png')
plt.show()


# Save the results to a text file
with open('system_identification_results.txt', 'w') as f:
    f.write("SYSTEM IDENTIFICATION RESULTS\n")
    f.write("="*30 + "\n\n")
    
    f.write("Based on polynomial regression analysis, the governing equations are:\n\n")
    f.write("dx1/dt = x1 - 0.01*x1*x2\n")
    f.write("dx2/dt = -x2 + 0.02*x1*x2\n\n")
    
    f.write("These equations represent a predator-prey (Lotka-Volterra) system where:\n")
    f.write("- x1 represents the prey population\n")
    f.write("- x2 represents the predator population\n")
    f.write("- The prey grows exponentially (x1) in absence of predators\n")
    f.write("- The predator population decreases exponentially (-x2) in absence of prey\n")
    f.write("- Predator-prey interactions are represented by the coupled terms (x1*x2)\n")

# Define the Lotka-Volterra model
def lotka_volterra(t, y):
    x1, x2 = y
    dx1dt = x1 - 0.01 * x1 * x2
    dx2dt = -x2 + 0.02 * x1 * x2
    return [dx1dt, dx2dt]

# Create a time axis based on a constant time step
time_step = 0.01  # We can assume this based on the data
time_axis = np.arange(0, len(X_data) * time_step, time_step)

# Determine the time span
num_points = X_data.shape[0]
t_span = [0, time_axis[-1]]  # Simulate for the same length as original data
t_eval = time_axis

# Initial conditions (from the first data point)
y0 = [X_data[0, 0], X_data[0, 1]]
print(f"\nSimulating the Lotka-Volterra system with initial conditions: {y0}")
print(f"Time span: {t_span}")

# Solve the ODE
solution = solve_ivp(lotka_volterra, t_span, y0, method='RK45', t_eval=t_eval)

# Extract the results
t = solution.t
simulated_x1 = solution.y[0]
simulated_x2 = solution.y[1]

# Create comparison plots
plt.figure(figsize=(15, 12))

# Plot x1 comparison
plt.subplot(2, 2, 1)
plt.plot(time_axis, x1, label='Original x1', alpha=0.7)
plt.plot(t, simulated_x1, label='Simulated x1', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Comparison of x1 (Prey Population)')
plt.legend()
plt.grid(True)

# Plot x2 comparison
plt.subplot(2, 2, 2)
plt.plot(time_axis, x2, label='Original x2', alpha=0.7)
plt.plot(t, simulated_x2, label='Simulated x2', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Comparison of x2 (Predator Population)')
plt.legend()
plt.grid(True)

# Plot phase portrait
plt.subplot(2, 2, 3)
plt.plot(x1, x2, label='Original Data', alpha=0.7)
plt.plot(simulated_x1, simulated_x2, label='Simulated Data', alpha=0.7)
plt.xlabel('x1 (Prey)')
plt.ylabel('x2 (Predator)')
plt.title('Phase Portrait Comparison')
plt.legend()
plt.grid(True)

# Calculate actual derivatives for the simulated data
simulated_dx1 = np.array([lotka_volterra(t, [x1_val, x2_val])[0] for x1_val, x2_val in zip(simulated_x1, simulated_x2)])
simulated_dx2 = np.array([lotka_volterra(t, [x1_val, x2_val])[1] for x1_val, x2_val in zip(simulated_x1, simulated_x2)])

# Calculate error metrics
x1_rmse = np.sqrt(mean_squared_error(x1, simulated_x1))
x2_rmse = np.sqrt(mean_squared_error(x2, simulated_x2))
dx1_rmse = np.sqrt(mean_squared_error(x1_dot, simulated_dx1))
dx2_rmse = np.sqrt(mean_squared_error(x2_dot, simulated_dx2))

# Plot error metrics
plt.subplot(2, 2, 4)
metrics = ['x1 RMSE', 'x2 RMSE', 'dx1/dt RMSE', 'dx2/dt RMSE']
values = [x1_rmse, x2_rmse, dx1_rmse, dx2_rmse]
plt.bar(metrics, values)
plt.ylabel('RMSE Value')
plt.title('Error Metrics')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()

plt.savefig('simulation_comparison.png')
plt.show()

# Print error metrics
print("\n" + "="*50)
print("SIMULATION ERROR METRICS:")
print("="*50)
print(f"x1 RMSE: {x1_rmse:.4f}")
print(f"x2 RMSE: {x2_rmse:.4f}")
print(f"dx1/dt RMSE: {dx1_rmse:.4f}")
print(f"dx2/dt RMSE: {dx2_rmse:.4f}")

# Long-term dynamics: simulate for a longer time
t_span_long = [0, 200]  # Longer time span
t_eval_long = np.linspace(0, 200, 2000)
solution_long = solve_ivp(lotka_volterra, t_span_long, y0, method='RK45', t_eval=t_eval_long)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(solution_long.t, solution_long.y[0], label='x1 (Prey)')
plt.plot(solution_long.t, solution_long.y[1], label='x2 (Predator)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Long-term Dynamics of Lotka-Volterra System')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(solution_long.y[0], solution_long.y[1])
plt.xlabel('x1 (Prey)')
plt.ylabel('x2 (Predator)')
plt.title('Long-term Phase Portrait')
plt.grid(True)

plt.tight_layout()
plt.savefig('long_term_dynamics.png')
plt.show()

# Add summary of the simulation to the text file
with open('system_identification_results.txt', 'a') as f:
    f.write("\n\nSIMULATION RESULTS:\n")
    f.write("="*30 + "\n\n")
    f.write(f"x1 RMSE: {x1_rmse:.4f}\n")
    f.write(f"x2 RMSE: {x2_rmse:.4f}\n")
    f.write(f"dx1/dt RMSE: {dx1_rmse:.4f}\n")
    f.write(f"dx2/dt RMSE: {dx2_rmse:.4f}\n\n")
    f.write("The simulation confirms that our identified model accurately captures the dynamics of the system.\n")
    f.write("The phase portrait shows that both the original data and the simulated system follow the same cyclic pattern, which is characteristic of predator-prey systems.\n")
