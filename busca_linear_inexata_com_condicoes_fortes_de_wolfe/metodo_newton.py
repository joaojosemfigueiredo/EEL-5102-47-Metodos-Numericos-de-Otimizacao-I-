import numpy as np
import sympy as sp
import pandas as pd
import time
from scipy.linalg import LinAlgError

# Definindo símbolos x e y
x, y = sp.symbols('x y')

# Escolha uma das funções de exemplo substituindo #user_function = ... pela função desejada
user_function = x**4 - 2*(x**2)*y + y**2 + x**2 - 2*x + 5
# Determinando as variáveis presentes na função do usuário
variables_in_function = [var for var in [x, y] if var in user_function.free_symbols]

# Compute gradient and Hessian
grad_f = [sp.diff(user_function, var) for var in [x, y]]
hess_f = sp.Matrix([[sp.diff(g, var) for var in [x, y]] for g in grad_f])

def verificar_autovalores_hessiana(hessiana_no_ponto):
    autovalores = np.linalg.eigvals(hessiana_no_ponto)
    print(f'autovalores {autovalores}')
    
    if all(val > 0 for val in autovalores):
        return "Matriz simétrica Definida Positiva (DP)"
    elif all(val >= 0 for val in autovalores):
        if any(val == 0 for val in autovalores):
            return "Matriz simétrica semidefinida positiva (SPD)"
    elif all(val < 0 for val in autovalores):
        return "Matriz simétrica Definida Negativa (DN)"
    else:
        return "Matriz indefinida"
    
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

def newton(problem, x_k, epsilon, max_iter, c1, c2, alpha_init):
    data = []  # Lista para coleta de dados de cada iteração
    iter = 0
    start_time = time.time()  # Marca o tempo inicial do processo

    while iter < max_iter:
        f_x_k = problem['f_x_k'](x_k)  # Avalia a função no ponto atual x_k
        grad = problem['full_grad'](x_k)  # Calcula o gradiente no ponto atual x_k
        hess = problem['full_hess'](x_k)  # Calcula a Hessiana no ponto atual x_k
        gradient_norm = np.linalg.norm(grad)  # Calcula a norma do gradiente

        if gradient_norm < epsilon:  # Verifica se a norma do gradiente é menor que epsilon
            print(f"Convergência alcançada na iteração {iter}. Norma do gradiente: {gradient_norm:.2e}")
            break

        classificacao = verificar_autovalores_hessiana(hess)
        print(classificacao)
        if classificacao == "Matriz simétrica Definida Positiva (DP)":
            p_k = -np.linalg.inv(hess) @ grad # Tenta resolver o sistema linear para encontrar a direção p_k
            print(f'Hessiana é invertível\n utilizando a direção de newton. p_k = {p_k}')
            
        else:
            print("Hessiana não é DP, utilizando a direção do gradiente.")
            p_k = -grad  # Usa a direção do gradiente se a Hessiana for singular
            print(f'p_k = {p_k}')

        alpha, _, _ = strong_wolfe_line_search(problem, p_k, x_k, c1, c2, alpha_init)  # Realiza a busca de linha
        x_k += alpha * p_k  # Atualiza o ponto x_k
        elapsed_time = time.time() - start_time  # Calcula o tempo decorrido desde o início

        # Coleta dados da iteração atual
        data.append([round(iter,3), round(x_k[0],3), round(x_k[1],3), round(p_k[0],3), round(p_k[1],3), round(gradient_norm,3), round(alpha,3), round(problem['f_x_k'](x_k),3), round(elapsed_time,3)])

        print(f"Iter {iter}: f_x_k = {f_x_k:.4f}, Norma do gradiente = {gradient_norm:.4f}, Alpha = {alpha:.4f}, Tempo = {elapsed_time:.4f} sec")

        iter += 1

    # Cria um DataFrame e salva em um arquivo Excel
    columns = ['Iteração', 'x_1', 'x_2', 'p_1', 'p_2', 'Norma do Gradiente', 'Alpha', 'f_x_k', 'Tempo Decorrido (s)']
    df = pd.DataFrame(data, columns=columns)
    filename = 'otimizacao_newton.xlsx'
    df.to_excel(filename, index=False)
    print(f"Resultados salvos em {filename}.")

    return x_k, problem['f_x_k'](x_k)

# Exemplo de uso
initial_point = np.array([1.0, 4.0])
max_iter = 1000
c1 = 0.1
c2 = 0.9
alpha_init = 1
epsilon = 1e-8
result, final_f_x_k = newton(problem, initial_point, epsilon, max_iter, c1, c2, alpha_init)
print(f"O ponto mínimo encontrado é: {result}")
print(f"O valor da função no ponto mínimo é: {final_f_x_k}")