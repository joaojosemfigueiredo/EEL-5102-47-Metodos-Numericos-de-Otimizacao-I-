import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar
import time

# Definindo símbolos x e y
x, y = sp.symbols('x y')

user_function = 10*x**2 + y**2

# Deriva a função em relação a x e y para obter o gradiente e a Hessiana
grad_f = [sp.diff(user_function, var) for var in [x, y]]
hess_f = sp.Matrix([[sp.diff(g, var) for var in [x, y]] for g in grad_f])

# Avalia o gradiente em um ponto específico
def evaluate_gradient(grad, values):
    subs_dict = dict(zip([x, y], values))
    return np.array([expr.subs(subs_dict) for expr in grad], dtype=float)

# Avalia o valor da função em um ponto específico
def evaluate_f_x_k(func, values):
    subs_dict = dict(zip([x, y], values))
    return float(func.subs(subs_dict))

# Avalia a Hessiana em um ponto específico
def evaluate_hessian(hess, values):
    subs_dict = dict(zip([x, y], values))
    return np.array(hess.subs(subs_dict)).astype(float)

# Adaptação do problema para utilizar as funções acima
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

# Método BFGS com busca linear exata
def bfgs(problem, x_k, max_iter, epsilon, c1, c2):
    n = len(x_k)
    B = np.eye(n)  # Inicializa a aproximação da matriz Hessiana como a identidade
    data = []

    for iter in range(max_iter):
        grad = problem['full_grad'](x_k)
        gradient_norm = np.linalg.norm(grad)

        if gradient_norm < epsilon:
            print("Convergência alcançada.")
            break

        p_k = -np.linalg.solve(B, grad)  # Direção de busca
        alpha_k = exact_line_search(problem, p_k, x_k)  # Encontra o tamanho do passo ótimo
        s = alpha_k * p_k
        x_new = x_k + s
        y = problem['full_grad'](x_new) - grad
        grad_x_k_1 = problem['full_grad'](x_new)
        print(f'grad_x_k_1 {grad_x_k_1}')
        print(f'grad x_k {grad}')
        print(f'y {y}')
        
        print(f"\nIteração {iter}")
        print(f"Ponto atual: {x_k}")
        print(f"Gradiente atual: {grad}")
        print(f"Norma do gradiente: {gradient_norm}")
        print(f"Direção de busca p_k: {p_k}")
        print(f"Tamanho do passo alpha_k: {alpha_k}")
        
        sy = np.dot(s, y)
        print(f"y_{iter}^T: {y}")
        print(f"s_{iter}: {s}")
        print(f"Produto Escalar y_{iter}^T s_{iter}: {sy}")
        
        if sy > 1e-10:
            rho = 1.0 / sy
            print(f"rho {rho}")
            V = np.eye(n) - rho * np.outer(s, y)
            B = V.T @ B @ V + rho * np.outer(s, s)

            print(f"I - rho dot(s,y^T) {V}")
            print(f"I - rho dot(y,s^T) {V.T}")

            print(f"Produto Externo s_{iter} s_{iter}^T Vezes RHO: \n{rho * np.outer(s, s)}")
            print(f"Aproximação da matriz Hessiana B_{iter+1}: \n{B}")

        x_k = x_new
        print(f"Novo ponto x_{iter+1}: {x_new}")
        print(f"Valor da função f(x_{iter+1}): {problem['f_x_k'](x_k)}")

    return x_k, problem['f_x_k'](x_k)

# Inicializa parâmetros
initial_point = np.array([0.1, 1])
max_iter = 3
c1 = 0.1
c2 = 0.9
epsilon = 1e-5
start_time = time.time()  # Marca o tempo inicial do processo

# Executa o BFGS
result, final_f_x_k = bfgs(problem, initial_point, max_iter, epsilon, c1, c2)
print(f"\nO ponto mínimo encontrado é: {result}")
print(f"O valor da função no ponto mínimo é: {final_f_x_k}")
