import numpy as np
import sympy as sp
import pandas as pd
import time

# Definindo símbolos x e y
x, y = sp.symbols('x y')

# Escolha uma das funções de exemplo substituindo #user_function = ... pela função desejada
user_function = x**4 - 2*(x**2)*y + y**2 + x**2 - 2*x + 5

# Compute gradient and Hessian
grad_f = [sp.diff(user_function, var) for var in [x, y]]
hess_f = sp.Matrix([[sp.diff(g, var) for var in [x, y]] for g in grad_f])

# Evaluate gradient at a point
def evaluate_gradient(grad, values):
    subs_dict = dict(zip([x, y], values))
    return np.array([expr.subs(subs_dict) for expr in grad], dtype=float)

# Evaluate cost at a point
def evaluate_f_x_k(func, values):
    subs_dict = dict(zip([x, y], values))
    return float(func.subs(subs_dict))

# Evaluate Hessian at a point
def evaluate_hessian(hess, values):
    subs_dict = dict(zip([x, y], values))
    return np.array(hess.subs(subs_dict)).astype(float)

# Adaptação do problema para utilizar as funções acima
problem = {
    "f_x_k": lambda point: evaluate_f_x_k(user_function, point),
    "full_grad": lambda point: evaluate_gradient(grad_f, point),
    "full_hess": lambda point: evaluate_hessian(hess_f, point)
}

def strong_wolfe_line_search(problem, d, x0, c1, c2,alpha_init):
    fx0 = problem["f_x_k"](x0)  # Avalia o custo no ponto inicial
    gx0 = problem["full_grad"](x0)  # Calcula o gradiente no ponto inicial
    max_iter = 3  # Máximo de iterações para a busca de linha
    alpham = 20  # Valor máximo para alpha
    alphap = 0  # Valor anterior de alpha
    gx0 = np.dot(gx0, d)  # Produto escalar do gradiente e direção de busca
    fxp = fx0  # Custa anterior
    i = 1  # Iterador

    while True:
        xx = x0 + alpha_init * d  # Calcula o novo ponto
        fxx = problem["f_x_k"](xx)  # Avalia o custo no novo ponto
        gxx = problem["full_grad"](xx)  # Calcula o gradiente no novo ponto
        fs = fxx  # Salva o custo para retorno
        gs = gxx  # Salva o gradiente para retorno
        gxx = np.dot(gxx, d)  # Produto escalar do gradiente e direção de busca

        # Verifica a primeira condição de Wolfe
        if (fxx > fx0 + c1 * alpha_init * gx0) or ((i > 1) and (fxx >= fxp)):
            alphas, fs, gs = zoom(problem, x0, d, alphap, alpha_init, fx0, gx0, c1, c2)
            return alphas, fs, gs

        # Verifica a segunda condição de Wolfe
        if abs(gxx) <= -c2 * gx0:
            return alpha_init, fs, gs

        # Verifica a condição de curvatura
        if gxx >= 0:
            alphas, fs, gs = zoom(problem, x0, d, alpha_init, alphap, fx0, gx0, c1, c2)
            return alphas, fs, gs

        alphap = alpha_init  # Atualiza alpha anterior
        fxp = fxx  # Atualiza custo anterior
        if i > max_iter:
            print("Máximo de iterações alcançado na busca de linha.")
            return alpha_init, fs, gs
        
        alpha_init = alpha_init + (alpham - alpha_init) * 0.8  # Atualiza alpha
        i += 1

def zoom(problem, x0, d, alphal, alphah, fx0, gx0, c1, c2):
    i = 0
    max_iter = 5

    while True:
        alpha_init = 0.5 * (alphal + alphah)  # Escolhe alpha como o ponto médio
        xx = x0 + alpha_init * d  # Calcula o novo ponto
        fxx = problem["f_x_k"](xx)  # Avalia o custo no novo ponto
        gxx = problem["full_grad"](xx)  # Calcula o gradiente no novo ponto
        fs = fxx  # Salva o custo para retorno
        gs = gxx  # Salva o gradiente para retorno
        gxx = np.dot(gxx, d)  # Produto escalar do gradiente e direção de busca
        xl = x0 + alphal * d  # Calcula o ponto com alphal
        fxl = problem["f_x_k"](xl)  # Avalia o custo no ponto com alphal

        if (fxx > fx0 + c1 * alpha_init * gx0) or (fxx >= fxl):
            alphah = alpha_init  # Atualiza alphah
        else:
            if abs(gxx) <= -c2 * gx0:
                return alpha_init, fs, gs
            if gxx * (alphah - alphal) >= 0:
                alphah = alphal  # Atualiza alphah se necessário
            alphal = alpha_init  # Atualiza alphal

        if i >= max_iter:
            print("Máximo de iterações alcançado no zoom.")
            return alpha_init, fs, gs
        i += 1


def gradient_descent_with_strong_wolfe(problem, x_k, c1, c2, epsilon, max_iter, alphax):
    data = []  # Lista para coletar os dados
    start_time = time.time()  # Marca o tempo inicial do processo

    for k in range(max_iter):
        grad_x_k = problem["full_grad"](x_k)  # Calcula o gradiente no ponto atual x_k
        norm_grad = np.linalg.norm(grad_x_k)  # Calcula a norma do gradiente
        if norm_grad <= epsilon:
            print(f"Convergência alcançada na iteração {k}.")
            break
        p_k = -grad_x_k  # Define a direção de busca como a negativa do gradiente
        alpha_k, fs, _ = strong_wolfe_line_search(problem, p_k, x_k, c1, c2, alphax)  # Realiza a busca de linha
        x_k_new = x_k + alpha_k * p_k  # Atualiza o ponto x_k
        elapsed_time = time.time() - start_time  # Calcula o tempo decorrido desde o início

        # Coleta dados incluindo o tempo decorrido para o DataFrame
        data.append([round(k,3), round(x_k[0],3), round(x_k[1],3), round(p_k[0],3), round(p_k[1],3), round(norm_grad,3), round(alpha_k,3), round(fs,3), round(elapsed_time,3)])
        x_k = x_k_new
    
    # Cria um DataFrame com os dados coletados
    columns = ['k', '(x_1)^k', '(x_2)^k', '(p_1)^k', '(p_2)^k', 'norma do gradiente', 'alpha_utilizado', 'f(x_k)', 'Tempo Decorrido (s)']
    df = pd.DataFrame(data, columns=columns)
    
    # Salva o DataFrame em um arquivo Excel
    filename = 'otimizacao_gradiente.xlsx'
    df.to_excel(filename, index=False)
    print(f"Resultado da otimização salvo em {filename}.")

    return x_k  # Retorna o ponto final de x_k

# Exemplo de execução
initial_point = np.array([1, 4])
epsilon = 1e-8
max_iter = 1000
c1 = 0.1
c2 = 0.9
alphax = 1

# Executando o método do gradiente
result = gradient_descent_with_strong_wolfe(problem, initial_point, c1, c2, epsilon, max_iter, alphax)
print(f"Resultado da otimização: x = {result}, f(x) = {problem['f_x_k'](result)}")