import numpy as np
import sympy as sp

# Definindo símbolos x e y
x, y = sp.symbols('x y')

# Função do usuário
#user_function = x**4 - 2*(x**2)*y + y**2 + x**2 - 2*x + 5 # f(x,y) = 4 com (x,y) = (1,1)
#user_function = 4*x**2 + 4*x*y + 3*y**2 + x + 2*y
#user_function = 2*x**2 + 6*x*y + y**2 + 4*x + 2*y
#user_function = x**3 - 12*x*y + 8*y**3
#user_function = (x**2 + y - 11)**2  + (x + y**2 - 7)**2 
#user_function = 10*(y-x**2)**2  + (1 - x)**2 
#user_function = x**4
user_function = 5*x**2 + y**2 + 4*x*y - 14*x - 6*y + 20

# Calculando o gradiente da função do usuário
grad_f = [sp.diff(user_function, var) for var in [x, y]]

# Criando uma função para avaliar o gradiente em um ponto específico
def evaluate_gradient(grad, values):
    subs_dict = dict(zip([x, y], values))
    return np.array([expr.subs(subs_dict) for expr in grad], dtype=float)

# Criando uma função para avaliar o custo (valor da função) em um ponto específico
def evaluate_cost(func, values):
    subs_dict = dict(zip([x, y], values))
    return float(func.subs(subs_dict))

# Adaptação do problema para utilizar as funções acima
problem = {
    "cost": lambda point: evaluate_cost(user_function, point),
    "full_grad": lambda point: evaluate_gradient(grad_f, point)
}

def strong_wolfe_line_search(problem, d, x0, c1, c2):
    fx0 = problem["cost"](x0)
    gx0 = problem["full_grad"](x0)
    
    max_iter = 3  # Máximo de iterações para a busca de linha
    alpham = 20  # Valor máximo para alpha
    alphap = 0  # Valor anterior de alpha
    alphax = 0.0866  # Valor inicial de alpha
    gx0 = np.dot(gx0, d)  # Produto escalar do gradiente e direção de busca
    fxp = fx0  # Custa anterior
    i = 1  # Iterador

    while True:
        xx = x0 + alphax * d
        fxx = problem["cost"](xx)
        gxx = problem["full_grad"](xx)
        fs = fxx
        gs = gxx
        gxx = np.dot(gxx, d)

        if (fxx > fx0 + c1 * alphax * gx0) or ((i > 1) and (fxx >= fxp)):
            alphas, fs, gs = zoom(problem, x0, d, alphap, alphax, fx0, gx0, c1, c2)
            return alphas, fs, gs

        if abs(gxx) <= -c2 * gx0:
            return alphax, fs, gs

        if gxx >= 0:
            alphas, fs, gs = zoom(problem, x0, d, alphax, alphap, fx0, gx0, c1, c2)
            return alphas, fs, gs

        alphap = alphax
        fxp = fxx

        if i > max_iter:
            return alphax, fs, gs
        
        r = 0.8  # A fixed value for simplicity
        alphax = alphax + (alpham - alphax) * r
        i += 1

def zoom(problem, x0, d, alphal, alphah, fx0, gx0, c1, c2):
    i = 0
    max_iter = 5

    while True:
        alphax = 0.5 * (alphal + alphah)
        xx = x0 + alphax * d
        fxx = problem["cost"](xx)
        gxx = problem["full_grad"](xx)
        fs = fxx
        gs = gxx
        gxx = np.dot(gxx, d)
        xl = x0 + alphal * d
        fxl = problem["cost"](xl)

        if (fxx > fx0 + c1 * alphax * gx0) or (fxx >= fxl):
            alphah = alphax
        else:
            if abs(gxx) <= -c2 * gx0:
                return alphax, fs, gs
            if gxx * (alphah - alphal) >= 0:
                alphah = alphal
            alphal = alphax

        i += 1
        if i > max_iter:
            return alphax, fs, gs

# Para usar as funções acima:
d = np.array([-1, -1])  # Direção de busca exemplo
x0 = np.array([0, 10])  # Ponto inicial exemplo
c1, c2 = 0.1, 0.9  # Constantes para as condições de Wolfe

# Realizando a busca de linha
alphas, fs, gs = strong_wolfe_line_search(problem, d, x0, c1, c2)

print(f"Passo ótimo: {alphas}, Valor da função: {fs}, Gradiente no novo ponto: {gs}")