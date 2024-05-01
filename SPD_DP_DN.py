import numpy as np
import sympy as sp

# Definindo símbolos x e y
x, y = sp.symbols('x y')

user_function = x**4 - 2*(x**2)*y + y**2 + x**2 - 2*x + 5 # f(x,y) = 4 com (x,y) = (1,1)
#user_function = 3*x**2 - 12*y - 2*(x-2*y)**3

# Determinando as variáveis presentes na função do usuário
variables_in_function = [var for var in [x, y] if var in user_function.free_symbols]

# Calculando o gradiente
grad_f = [sp.diff(user_function, var) for var in variables_in_function]

# Encontrando os pontos críticos
critical_points = sp.solve(grad_f, variables_in_function)

# Preparando para processar os pontos críticos de acordo com o número de variáveis
real_critical_points = []

# Verificando o tipo de retorno e processando de acordo
if isinstance(critical_points, dict):
    if all(sp.im(value) == 0 for value in critical_points.values()):
        real_critical_points.append(tuple(critical_points[var] for var in variables_in_function))
elif isinstance(critical_points, (set, list)):
    real_critical_points = [tuple_ for tuple_ in critical_points if all(sp.im(val) == 0 for val in tuple_)]

print("Pontos críticos reais:")
print(real_critical_points)

# Calculando a matriz Hessiana
hessian = sp.hessian(user_function, variables_in_function)

def verificar_autovalores_hessiana(ponto, hessian):
    substituicoes = dict(zip(variables_in_function, ponto))
    print(f'substituicoes {substituicoes}')
    hessiana_no_ponto = hessian.subs(substituicoes)
    print(f'hessiana_no_ponto {hessiana_no_ponto}')
    autovalores = np.linalg.eigvals(np.array(hessiana_no_ponto).astype(np.float64))
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

# Função para verificar a convexidade baseada na matriz Hessiana
def verificar_convexidade(hessian, variables):
    # Escolha de um ponto arbitrário para análise
    ponto_arbitrario = {var: 2 for var in variables}  # Exemplo: {x: 0.5, y: 0.5}
    print(f'ponto_arbitrario {ponto_arbitrario}')
    hessiana_no_ponto = hessian.subs(ponto_arbitrario)
    print(f'hessiana_no_ponto {hessiana_no_ponto}')
    autovalores = np.linalg.eigvals(np.array(hessiana_no_ponto).astype(np.float64))
    print(f'autovalores {autovalores}')
    
    if all(val >= 0 for val in autovalores):
        for val in autovalores:
            print(val)
        print("A função pode ser convexa, baseando-se na análise de um ponto arbitrário.")
    else:
        print("A função não é convexa, baseando-se na análise de um ponto arbitrário. Isso indica que a matriz Hessiana não é semidefinida positiva em todo o domínio.")


# Aplicando a verificação para cada ponto crítico real
for ponto in real_critical_points:
    classificacao = verificar_autovalores_hessiana(ponto, hessian)
    print(f"Ponto {ponto}: {classificacao}")

print(f'hessian {hessian}')
# Executando a verificação de convexidade
verificar_convexidade(hessian, variables_in_function)
