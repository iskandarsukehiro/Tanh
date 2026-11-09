import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Generic Tanh", layout="wide")
st.title("3. Hyperbolic Tangent (Tanh)")


st.sidebar.header("Neuron Parameters")
x1 = st.sidebar.number_input("Input 1 (x1)", value=1.0)
w1 = st.sidebar.number_input("Weight 1 (w1)", value=0.5)
x2 = st.sidebar.number_input("Input 2 (x2)", value=-1.0)
w2 = st.sidebar.number_input("Weight 2 (w2)", value=0.8)
x3 = st.sidebar.number_input("Input 3 (x3)", value=0.5)
w3 = st.sidebar.number_input("Weight 3 (w3)", value=-0.2)
bias = st.sidebar.number_input("Bias (b)", value=0.1)

st.subheader("1. Neuron Inputs & Weights")
df = pd.DataFrame({
    "Input (x)": [x1, x2, x3],
    "Weight (w)": [w1, w2, w3]
}, index=["Node 1", "Node 2", "Node 3"])
st.table(df)


st.subheader("2. Weighted Sum Calculation (z)")
z = (x1 * w1) + (x2 * w2) + (x3 * w3) + bias
st.latex(r"z = (x_1 \times w_1) + (x_2 \times w_2) + (x_3 \times w_3) + b")
st.latex(f"z = ({x1} \\times {w1}) + ({x2} \\times {w2}) + ({x3} \\times {w3}) + {bias} = {z:.2f}")


st.subheader("3. Activation Output")
activation_output = np.tanh(z)
st.success(f"**Tanh Output:** tanh({z:.2f}) = **{activation_output:.4f}**")


st.subheader("4. Function Visualisation")
xgrid = np.linspace(-5, 5, 400)
ygrid = np.tanh(xgrid)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(xgrid, ygrid, label="Tanh Curve", color="purple", linewidth=2)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.plot(z, activation_output, 'ro', markersize=8, label=f"Result: f({z:.2f}) = {activation_output:.4f}")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)