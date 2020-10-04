# coef_vec = [1, -1, -1, 1]
coef_vec = [1, 0, -1]

p_prev = 0.5
p_next = 100

while True:

    b = [0]*len(coef_vec)
    c = [0]*len(coef_vec)
    b[0] = coef_vec[0]
    c[0] = coef_vec[0]
    for i in range(1, len(coef_vec)):
        b[i] = coef_vec[i] + p_prev*b[i-1]
        c[i] = b[i] + p_prev*c[i-1]

    p_next = p_prev - b[len(coef_vec)-1]/c[len(coef_vec)-2]
    if abs(p_next - p_prev) < 1e-8:
        break
    else:
        p_prev = p_next

print(p_next)
