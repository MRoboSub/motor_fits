import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot(x_data, y_data, z_expected, z_fit, x_label, y_label, z_label):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x_data, y_data, z_expected, marker="o")
    ax.scatter(x_data, y_data, z_fit, marker="^")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.show()


def plot2(x, y, fit, x_label, y_label, z_label):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    z = make_matrix(x, y) @ fit
    x = x[z <= 1.0]
    y = y[z <= 1.0]
    z = z[z <= 1.0]
    ax.scatter(x, y, z, marker="o")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.show()


def make_matrix(a, b) -> np.ndarray:
    return np.vstack(
        (
            np.ones(a.shape),
            a,
            b,
            a**2,
            a * b,
            b**2,
            a**3,
            a**2 * b,
            a * b**2,
            b**3,
        )
    ).T


def fit_table(table, input_a: str, input_b: str, output: str):
    a_arr, b_arr, out_arr = table.reset_index()[[input_a, input_b, output]].to_numpy().T
    arr = make_matrix(a_arr, b_arr)
    fit = np.linalg.lstsq(arr, out_arr)[0]
    calc = arr @ fit
    plot(a_arr, b_arr, out_arr, calc, input_a, input_b, output)
    return fit


def force_forward(table):
    forward = table.drop(table[table["Force"] <= 0].index)
    backward = table.drop(table[table["Force"] >= 0].index)
    f_force_fit = fit_table(forward, "Output", "Voltage", "Force")
    b_force_fit = fit_table(backward, "Output", "Voltage", "Force")
    return f_force_fit, b_force_fit


def force_inverse(table):
    forward = table.drop(table[table["Force"] <= 0].index)
    backward = table.drop(table[table["Force"] >= 0].index)
    f_pwm_fit = fit_table(forward, "Force", "Voltage", "Output")
    b_pwm_fit = fit_table(backward, "Force", "Voltage", "Output")
    return f_pwm_fit, b_pwm_fit


def current_forward(table):
    forward = table.drop(table[table["Force"] <= 0].index)
    backward = table.drop(table[table["Force"] >= 0].index)
    f_current_fit = fit_table(forward, "Output", "Voltage", "Current")
    b_current_fit = fit_table(backward, "Output", "Voltage", "Current")
    return f_current_fit, b_current_fit


if __name__ == "__main__":
    data = []
    for i in range(10, 22, 2):
        df = pd.read_csv(f"{i}V.tsv", sep="\t").set_axis(
            ["PWM", "RPM", "Current", "Voltage", "Power", "Force", "Efficiency"],
            axis="columns",
        )
        df["Output"] = (df["PWM"] - 1500) / 400
        df.set_index(["PWM", "Voltage"])
        data.append(df)
    table = pd.concat(data)
    f_power, b_power = force_forward(table)
    f_force, b_force = force_inverse(table)
    f_current, b_current = current_forward(table)
    print(f"{f_power=!r}")
    print(f"{b_power=!r}")
    print(f"{f_force=!r}")
    print(f"{b_force=!r}")
    print(f"{f_current=!r}")
    print(f"{b_current=!r}")
    force = np.linspace(0, 7, 100)
    voltage = np.linspace(10, 20, 100)
    f, v = np.meshgrid(force, voltage)
    plot2(f.flatten(), v.flatten(), f_force, "force", "voltage", "output")
