import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar
import time

# Definindo símbolos x e y
x, y = sp.symbols('x y')

# Função de usuário
user_function = 5*x**2 + y**2 + 4*x*y - 14*x - 6*y + 20

# Deriva a função para obter o gradiente e a Hessiana
grad_f = [sp.diff(user_function, var) for var in [x, y]]

# Avalia o gradiente em um ponto específico
def evaluate_gradient(grad, values):
    subs_dict = dict(zip([x, y], values))
    return np.array([expr.subs(subs_dict) for expr in grad], dtype=float)

# Avalia o valor da função em um ponto específico
def evaluate_f_x_k(func, values):
    subs_dict = dict(zip([x, y], values))
    return float(func.subs(subs_dict))

# Adaptação do problema para usar funções acima
problem = {
    "f_x_k": lambda point: evaluate_f_x_k(user_function, point),
    "full_grad": lambda point: evaluate_gradient(grad_f, point),
}

# Busca linear exata
def exact_line_search(problem, p_k, x_k):
    def f_alpha(alpha):
        return problem["f_x_k"](x_k + alpha * p_k)
    
    res = minimize_scalar(f_alpha)
    return res.x

# Método de gradiente com busca linear exata
def gradient_method(problem, x_k, max_iter, epsilon):
    data = []

    for iter in range(max_iter):
        grad = problem['full_grad'](x_k)
        gradient_norm = np.linalg.norm(grad)

        if gradient_norm < epsilon:
            break

        print(f"\nIteração {iter}")
        print(f"Ponto atual: {x_k}")
        p_k = -grad  # Direção de descida
        alpha_k = exact_line_search(problem, p_k, x_k)  # Encontra o tamanho do passo ótimo
        x_k += alpha_k * p_k

        # Coleta de dados para essa iteração
        data.append({
            "iter": iter,
            "x_k": x_k.tolist(),
            "gradient_norm": gradient_norm,
            "alpha_k": alpha_k,
            "f_x_k": problem['f_x_k'](x_k)
        })
        
        print(f"Ponto atual: {x_k}")
        print(f"Gradiente atual: {grad}")
        print(f"Norma do gradiente: {gradient_norm}")
        print(f"Direção de busca p_k: {p_k}")
        print(f"Tamanho do passo alpha_k: {alpha_k}")
        print(f"Valor da função f(x_k): {problem['f_x_k'](x_k)}")
        
    return x_k, problem['f_x_k'](x_k), data

# Executa método de gradiente
initial_point = np.array([0.0, 10.0])
max_iter = 3
epsilon = 1e-8

result, final_f_x_k, data = gradient_method(problem, initial_point, max_iter, epsilon)
print(f"\nO ponto mínimo encontrado é: {result}")
print(f"O valor da função no ponto mínimo é: {final_f_x_k}")
