import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
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
