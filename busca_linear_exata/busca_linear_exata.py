import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar

# Definindo símbolos x e y
x, y = sp.symbols('x y')

# Definição da função do usuário
user_function = (x-1)**2 + y**2 + x

# Calcula gradiente e Hessiana
grad_f = [sp.diff(user_function, var) for var in [x, y]]
hess_f = sp.Matrix([[sp.diff(g, var) for var in [x, y]] for g in grad_f])

# Avalia gradiente em um ponto específico
def evaluate_gradient(grad, values):
    subs_dict = dict(zip([x, y], values))
    return np.array([float(expr.subs(subs_dict)) for expr in grad])

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
        x_alpha = np.array(x_k) + alpha * np.array(p_k)
        return problem["f_x_k"](x_alpha.tolist())
    
    res = minimize_scalar(f_alpha)
    alpha_exato = res.x  # O valor de alpha que minimiza a função f_alpha
    print(f"\nAlpha exato encontrado: {alpha_exato:.4f}")
    return alpha_exato

x_k = [3, 0]
d_k = [1, 0]
alpha_k = exact_line_search(problem, d_k, x_k)  # Busca linear exata para tamanho do passo
