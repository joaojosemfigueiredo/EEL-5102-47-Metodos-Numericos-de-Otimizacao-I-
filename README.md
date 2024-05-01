# EEL-5102-47-Métodos Numéricos de Otimização I

Este repositório contém os códigos desenvolvidos como parte do trabalho de conclusão da disciplina EEL 5102-47 – Métodos Numéricos de Otimização I, oferecida pelo Programa de Pós-Graduação em Engenharia Elétrica da Universidade Federal de Santa Catarina (UFSC).

## Objetivo

O objetivo deste trabalho é explorar diversos métodos de otimização, aplicando conceitos teóricos em problemas práticos através da implementação de algoritmos numéricos. Cada script demonstra uma técnica específica de otimização, facilitando o entendimento de suas aplicações e limitações.

## Descrições dos Scripts

Aqui estão descrições breves de cada um dos nove scripts incluídos neste repositório, detalhando suas funções e a metodologia aplicada.

### Código 1: Otimização com Busca Linear Exata (busca_linear_exata.py)

Este script Python realiza uma otimização utilizando métodos de gradiente e Hessiana. Ele define uma função matemática, calcula seu gradiente e matriz Hessiana, e utiliza esses cálculos para executar uma busca linear exata. O objetivo é encontrar o passo ótimo (`alpha`) que minimiza a função em uma direção dada. Este exemplo específico usa a função \((x-1)^2 + y^2 + x\) para demonstrar como calcular o gradiente e a Hessiana, e como esses podem ser usados para otimizar a função em um ponto específico. Este código pode ser útil para estudantes e profissionais que desejam entender melhor as técnicas de otimização numérica aplicadas a funções de várias variáveis.

### Código 2: Método de Gradiente com Busca Linear (metodo_gradiente_exato.py)

Este script implementa o método de gradiente para encontrar o mínimo de uma função de duas variáveis utilizando uma busca linear exata para determinar o passo ótimo. A função em questão é \(5x^2 + y^2 + 4xy - 14x - 6y + 20\), e o script calcula o gradiente e o utiliza para mover-se na direção oposta ao gradiente (direção de descida) a partir de um ponto inicial. Utiliza critérios de parada baseados na norma do gradiente e um número máximo de iterações. Este exemplo é ideal para ilustrar a aplicação do método de gradiente em funções quadráticas e pode servir como ferramenta educativa para entender melhor como os algoritmos de otimização funcionam na prática.

### Código 3: Método de Newton para Otimização (metodo_newton_exato.py)

Este script implementa o método de Newton para otimização, aplicado a uma função de duas variáveis \(10x^2 + y^2\). O script calcula o gradiente e a matriz Hessiana da função, e utiliza esses cálculos para iterativamente ajustar a posição em busca do ponto mínimo. Um destaque do código é a verificação do tipo de matriz Hessiana (definida positiva, semidefinida positiva, definida negativa ou indefinida) para decidir a direção de descida. Se a Hessiana é definida positiva, o método segue a direção de Newton; caso contrário, utiliza-se a direção do gradiente. A busca linear exata é usada para encontrar o tamanho do passo ideal em cada iteração. Este código é uma excelente ferramenta de aprendizado para entender o método de Newton e suas aplicações em problemas de otimização.

### Código 4: Método BFGS para Otimização com Busca Linear Exata (quasi_newton_exato.py)

Este script implementa o método BFGS, uma técnica de otimização quase-Newton para encontrar o mínimo local de uma função sem precisar calcular a Hessiana. A função exemplo é \(10x^2 + y^2\). O script realiza o cálculo do gradiente e usa uma aproximação inicial da Hessiana como a matriz identidade. Em cada iteração, ajusta-se essa aproximação com base nos gradientes e direções de busca observados, aplicando correções para manter as propriedades da matriz. Utiliza-se uma busca linear exata para determinar o tamanho do passo. Este código é uma excelente escolha para visualizar como métodos quase-Newton como o BFGS podem ser eficientes para otimização em problemas de larga escala.

### Código 5: Busca de Linha com as Condições de Strong Wolfe (backtracking_line_search.py)

Este script implementa a busca de linha com as condições de Strong Wolfe, um método fundamental em otimização para encontrar um tamanho de passo adequado que respeite as condições de Wolfe para descida. A função objetivo é \(5x^2 + y^2 + 4xy - 14x - 6y + 20\). O algoritmo usa a avaliação do gradiente e do custo da função para iterativamente ajustar o tamanho do passo até que as condições de Wolfe sejam satisfeitas. Este método ajuda a garantir convergência adequada em algoritmos de otimização, como métodos de gradiente conjugado ou BFGS, ao escolher um passo que não seja nem muito pequeno nem excessivamente grande. Este código é especialmente útil para quem estuda métodos de otimização e deseja implementar um controle rigoroso sobre a escolha do tamanho do passo.


### Código 6: Método de Descida de Gradiente com Condições de Strong Wolfe (metodo_gradiente.py)

Este script realiza um método de descida de gradiente aplicando as condições de Strong Wolfe para a escolha do tamanho do passo, otimizando a função \(x^4 - 2x^2y + y^2 + x^2 - 2x + 5\). As condições de Strong Wolfe garantem a adequação do passo tomado em cada iteração, promovendo a eficácia e a eficiência da convergência do método. Durante o processo, os dados de cada iteração são coletados, incluindo tempo decorrido, e salvos em um arquivo Excel, permitindo uma análise detalhada do desempenho do método. Esse código é ideal para pesquisadores e engenheiros que desejam aplicar e entender profundamente técnicas avançadas de otimização em problemas complexos.

### Código 7: Método de Newton com Condições de Strong Wolfe (metodo_newton.py)

Este script aplica o método de Newton para otimização, utilizando condições de Strong Wolfe para a busca de linha, focando na função \(x^4 - 2x^2y + y^2 + x^2 - 2x + 5\). O método de Newton utiliza a Hessiana da função e o gradiente para determinar a direção de busca e o tamanho do passo. A validação da Hessiana como definida positiva é crucial para garantir que o método siga uma direção que efetivamente reduza a função objetivo. O script também inclui um tratamento de exceções para Hessiana não invertível, utilizando a direção do gradiente como alternativa. O progresso da otimização é detalhadamente registrado e salvo em um arquivo Excel, proporcionando uma ferramenta útil para análise de performance e convergência do algoritmo em problemas complexos de otimização.

### Código 8: Método BFGS Quasi-Newton com Busca de Linha Strong Wolfe (metodo_quasi_newton.py)

Este script implementa o método BFGS, uma variação eficiente do método Quasi-Newton para otimização, aplicado à função \(x^4 - 2x^2y + y^2 + x^2 - 2x + 5\). Ele utiliza uma aproximação inicial da matriz Hessiana inversa e atualiza essa aproximação em cada iteração usando as informações dos gradientes e das direções de busca. O processo inclui uma busca de linha Strong Wolfe para garantir que o tamanho do passo seja adequado, otimizando a eficiência da convergência. Além disso, todos os dados de iteração, incluindo tempo decorrido, são coletados e salvos em um arquivo Excel, permitindo uma análise detalhada e compreensiva do processo de otimização. Este código é ideal para quem estuda métodos avançados de otimização e deseja uma implementação prática e eficaz em problemas complexos.

### Código 9: Análise de Pontos Críticos e Verificação de Convexidade (SPD_DP_DN.py)

Este script Python utiliza o módulo SymPy para realizar uma análise detalhada dos pontos críticos e verificar a convexidade da função \(x^4 - 2x^2y + y^2 + x^2 - 2x + 5\). Ele calcula o gradiente e a matriz Hessiana da função, resolve as equações do gradiente para encontrar os pontos críticos e avalia se estes são mínimos, máximos ou pontos de sela. Além disso, o script verifica a convexidade da função ao analisar os autovalores da matriz Hessiana em um ponto arbitrário. Esta análise fornece insights valiosos sobre a natureza da função, úteis para aplicações em otimização e em teoria de funções de várias variáveis. O script também destaca a implementação de conceitos matemáticos avançados em programação para análise simbólica e numérica.

## Uso

Para executar qualquer um dos scripts, é necessário ter Python instalado em sua máquina, juntamente com as bibliotecas NumPy e SymPy. Os scripts podem ser executados individualmente, conforme a necessidade de explorar cada método de otimização.

## Contribuições

Sugestões e melhorias são bem-vindas através de pull requests.

## Licença

Este projeto é licenciado sob a Licença MIT - veja o arquivo LICENSE.md para mais detalhes.

## Contato

Para mais informações, entre em contato com [João José Medeiros de Figueiredo] através do e-mail [joaojose.mfigueiredo@gmail.com].

