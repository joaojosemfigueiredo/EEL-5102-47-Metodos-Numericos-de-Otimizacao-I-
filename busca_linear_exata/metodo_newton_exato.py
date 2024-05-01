import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar
import time

# Definindo símbolos x e y
x, y = sp.symbols('x y')

user_function = 10*x**2 + y**2 

# Calcula gradiente e Hessiana
grad_f = [sp.diff(user_function, var) for var in [x, y]]
hess_f = sp.Matrix([[sp.diff(g, var) for var in [x, y]] for g in grad_f])

# Avalia gradiente em um ponto específico
def evaluate_gradient(grad, values):
    subs_dict = dict(zip([x, y], values))
    return np.array([expr.subs(subs_dict) for expr in grad], dtype=float)

# Avalia valor da função em um ponto específico
def evaluate_f_x_k(func, values):
    subs_dict = dict(zip([x, y], values))
    return float(func.subs(subs_dict))

# Avalia Hessiana em um ponto específico
def evaluate_hessian(hess, values):
    subs_dict = dict(zip([x, y], values))
    return np.array(hess.subs(subs_dict)).astype(float)

# Adaptação do problema para usar funções acima
problem = {
    "f_x_k": lambda point: evaluate_f_x_k(user_function, point),
    "full_grad": lambda point: evaluate_gradient(grad_f, point),
    "full_hess": lambda point: evaluate_hessian(hess_f, point)
}

# Busca linear exata
def exact_line_search(problem, p_k, x_k):
    def f_alpha(alpha):
        return problem["f_x_k"](x_k + alpha * p_k)
    
    res = minimize_scalar(f_alpha)
    alpha_exato = res.x  # O valor de alpha que minimiza a função f_alpha
    print(f"\nAlpha exato encontrado: {alpha_exato:.4f}")
    return alpha_exato

# Método de Newton
def newton_method(problem, x_k, max_iter, epsilon):
    for iter in range(max_iter):
        grad = problem['full_grad'](x_k)
        hess = problem['full_hess'](x_k)
        gradient_norm = np.linalg.norm(grad)

        if gradient_norm < epsilon:
            print("Convergência alcançada.")
            break

        try:
            d_k = -np.linalg.solve(hess, grad) # Tenta resolver o sistema linear para encontrar a direção p_k
        except np.linalg.LinAlgError:
            print("Hessiana não é invertível, usando direção de gradiente.")
            d_k = -grad # Usa a direção do gradiente se a Hessiana for singular
        
        alpha_k = exact_line_search(problem, d_k, x_k)  # Busca linear exata para tamanho do passo
        x_k += alpha_k * d_k
        
        print(f"\nIteração {iter}")
        print(f"Ponto atual: {x_k}")
        print(f"Gradiente atual: {grad}")
        print(f"Norma do gradiente: {gradient_norm}")
        print(f"Direção de busca d_k: {d_k}")
        print(f"Tamanho do passo alpha_k: {alpha_k}")

    return x_k, problem['f_x_k'](x_k)

# Executa método de Newton
initial_point = np.array([0.1, 1])
max_iter = 3
epsilon = 1e-5

result, final_f_x_k = newton_method(problem, initial_point, max_iter, epsilon)
print(f"\nO ponto mínimo encontrado é: {result}")
print(f"O valor da função no ponto mínimo é: {final_f_x_k}")
