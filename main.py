import numpy as np
from prettytable import PrettyTable


def calculate_m(e, v):
    return round(e / (2 * (1 + v)), 3)


def calculate_l(e, v):
    return round(e * v / ((1 + v) * (1 - 2 * v)), 3)


def calculate_matrix_materials(m, l):
    sum_m_l = round(l + 2 * m, 3)
    row1 = [sum_m_l, l, l, 0, 0, 0]
    row2 = [l, sum_m_l, l, 0, 0, 0]
    row3 = [l, l, sum_m_l, 0, 0, 0]
    row4 = [0, 0, 0, m, 0, 0]
    row5 = [0, 0, 0, 0, m, 0]
    row6 = [0, 0, 0, 0, 0, m]
    return np.array([row1, row2, row3, row4, row5, row6])


def calculate_matrix_s(m, l):
    a = round(((3*l+2*m)/(3*(l+2*m))), 3)
    b = round(((2*(3*l+8*m))/(15*(l+2*m))), 3)
    sub_a_b = round(((a-b)/3), 3)
    sum_a_b = round(((a+2*b)/3), 3)
    row1 = [sum_a_b, sub_a_b, sub_a_b, 0, 0, 0]
    row2 = [sub_a_b, sum_a_b, sub_a_b, 0, 0, 0]
    row3 = [sub_a_b, sub_a_b, sum_a_b, 0, 0, 0]
    row4 = [0, 0, 0, b, 0, 0]
    row5 = [0, 0, 0, 0, b, 0]
    row6 = [0, 0, 0, 0, 0, b]
    return np.array([row1, row2, row3, row4, row5, row6])


def print_matrix(matrix, label):
    p = PrettyTable()
    for row in matrix:
        p.add_row(row)

    print(label)
    print(p.get_string(header=False, border=True))


# Matrix materials
e_m = 4.6
v_m = 0.36

# Particle materials
e_i = 86.8
v_i = 0.23

m_m = calculate_m(e_m, v_m)
m_i = calculate_m(e_i, v_i)
l_m = calculate_l(e_m, v_m)
l_i = calculate_l(e_i, v_i)

identity_matrix = np.identity(6)
c_m = calculate_matrix_materials(m_m, l_m)
c_i = calculate_matrix_materials(m_i, l_i)
s_m = calculate_matrix_s(m_m, l_m)

print('Lambda m: ', l_m)
print('Mu m: ', m_m)
print('Lambda i: ', l_i)
print('Mu m: ', m_i)
print_matrix(c_m, 'C_m')
print_matrix(c_i, 'C_i')
print_matrix(s_m, 'S_m')

index = float(input('Enter index (0-1)'))
if index < 0 or index > 1:
    raise 'Index must be within 0-1'

# First task
c_mix = np.add(c_m, np.multiply(index, np.subtract(c_i, c_m)))
print_matrix(c_mix, 'C_mix')

# Second task
mult_s_m = np.dot(s_m, c_m)
ci_sub_cm = np.subtract(c_i, c_m)
mult_sm_divide_ci_sub_cm = np.dot(mult_s_m, np.linalg.inv(ci_sub_cm))
right_part = np.add(identity_matrix, mult_sm_divide_ci_sub_cm)
c_dd = np.dot(c_mix, right_part)
print_matrix(c_dd, 'C_dd')

# Third task
sm_index = np.multiply(1 - index, s_m)
mult_s_m = np.dot(sm_index, c_m)
ci_sub_cm = np.subtract(c_i, c_m)
mult_sm_divide_ci_sub_cm = np.dot(mult_s_m, np.linalg.inv(ci_sub_cm))
right_part = np.add(identity_matrix, mult_sm_divide_ci_sub_cm)
c_mt = np.dot(c_mix, right_part)
print_matrix(c_mt, 'C_mt')
