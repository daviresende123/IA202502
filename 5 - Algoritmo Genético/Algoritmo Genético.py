import numpy as np
import random

def funcao_adequacao(x):
    """Função objetivo: f(x) = x^2 - 3x + 4"""
    return x**2 - 3*x + 4

def binario_para_decimal(vetor_binario, x_min, x_max):
    """Converte vetor binário para valor decimal no intervalo [x_min, x_max]"""
    decimal = int(''.join(map(str, vetor_binario)), 2)
    return x_min + (x_max - x_min) * (decimal / (2**len(vetor_binario) - 1))

def criar_populacao_inicial(tamanho_populacao, num_bits):
    """Cria população inicial aleatória"""
    return [np.random.randint(2, size=num_bits).tolist() for _ in range(tamanho_populacao)]

def selecao_por_torneio(populacao, valores_adequacao, tamanho_torneio):
    """Seleção por torneio - retorna o melhor indivíduo do torneio"""
    selecionados = np.random.choice(len(populacao), tamanho_torneio, replace=False)
    melhor = selecionados[np.argmax([valores_adequacao[i] for i in selecionados])]
    return populacao[melhor].copy()

def crossover(pai1, pai2, taxa_crossover):
    """Crossover de um ponto"""
    if random.random() < taxa_crossover:
        ponto = random.randint(1, len(pai1) - 1)
        filho1 = pai1[:ponto] + pai2[ponto:]
        filho2 = pai2[:ponto] + pai1[ponto:]
        return filho1, filho2
    return pai1.copy(), pai2.copy()

def mutacao(individuo, taxa_mutacao):
    """Mutação bit-flip"""
    return [bit if random.random() > taxa_mutacao else 1 - bit for bit in individuo]

def algoritmo_genetico(tamanho_populacao, num_bits, x_min, x_max, num_geracoes, 
                       taxa_crossover, taxa_mutacao, tamanho_torneio, elitismo=True):
    """
    Algoritmo Genético para maximizar f(x) = x^2 - 3x + 4
    
    Parâmetros:
    - tamanho_populacao: número de indivíduos (4 a 30)
    - num_bits: tamanho da representação binária
    - x_min, x_max: intervalo de busca
    - num_geracoes: número de gerações (5 a 20)
    - taxa_crossover: probabilidade de crossover (0.7)
    - taxa_mutacao: probabilidade de mutação (0.01)
    - tamanho_torneio: tamanho do torneio para seleção
    - elitismo: se True, preserva o melhor indivíduo
    """
    populacao = criar_populacao_inicial(tamanho_populacao, num_bits)
    melhor_individuo = None
    melhor_adequacao = float('-inf')
    
    historico = []

    for geracao in range(num_geracoes):
        # Avaliar população
        valores_adequacao = [funcao_adequacao(binario_para_decimal(individuo, x_min, x_max)) 
                            for individuo in populacao]
        
        # Atualizar melhor indivíduo
        indice_melhor = np.argmax(valores_adequacao)
        if valores_adequacao[indice_melhor] > melhor_adequacao:
            melhor_adequacao = valores_adequacao[indice_melhor]
            melhor_individuo = populacao[indice_melhor].copy()
        
        historico.append({
            'geracao': geracao + 1,
            'melhor_adequacao': melhor_adequacao,
            'media_adequacao': np.mean(valores_adequacao),
            'melhor_x': binario_para_decimal(melhor_individuo, x_min, x_max)
        })

        # Criar nova população
        nova_populacao = []
        
        # Elitismo: preservar o melhor indivíduo
        if elitismo:
            nova_populacao.append(melhor_individuo.copy())
        
        # Gerar o restante da população
        while len(nova_populacao) < tamanho_populacao:
            pai1 = selecao_por_torneio(populacao, valores_adequacao, tamanho_torneio)
            pai2 = selecao_por_torneio(populacao, valores_adequacao, tamanho_torneio)
            filho1, filho2 = crossover(pai1, pai2, taxa_crossover)
            nova_populacao.append(mutacao(filho1, taxa_mutacao))
            if len(nova_populacao) < tamanho_populacao:
                nova_populacao.append(mutacao(filho2, taxa_mutacao))
        
        populacao = nova_populacao[:tamanho_populacao]  # Garantir tamanho exato

        print(f"Geração {geracao + 1}: Melhor Adequação = {melhor_adequacao:.4f}, "
              f"x = {binario_para_decimal(melhor_individuo, x_min, x_max):.4f}")

    return binario_para_decimal(melhor_individuo, x_min, x_max), melhor_adequacao, historico


# ========== CONFIGURAÇÃO INICIAL (REQUISITOS DO EXERCÍCIO) ==========
num_bits = 10
x_min = -10
x_max = 10
taxa_crossover = 0.7
taxa_mutacao = 0.01
tamanho_torneio = 2

print("="*70)
print("ALGORITMO GENÉTICO - Maximização de f(x) = x² - 3x + 4")
print("="*70)

# ========== TESTES COM DIFERENTES CONFIGURAÇÕES ==========
parametros = [
    {"tamanho_populacao": 4, "num_geracoes": 5},   # Configuração mínima do exercício
    {"tamanho_populacao": 10, "num_geracoes": 5},
    {"tamanho_populacao": 20, "num_geracoes": 10},
    {"tamanho_populacao": 30, "num_geracoes": 20},
    {"tamanho_populacao": 30, "num_geracoes": 5},
    {"tamanho_populacao": 15, "num_geracoes": 15},
    {"tamanho_populacao": 30, "num_geracoes": 10},
    {"tamanho_populacao": 20, "num_geracoes": 20},
    {"tamanho_populacao": 10, "num_geracoes": 20},
]

print("\n>>> TESTANDO DIFERENTES CONFIGURAÇÕES <<<\n")

resultados = []
for i, params in enumerate(parametros, 1):
    print(f"\n[TESTE {i}] Parâmetros: {params}")
    print("-" * 70)
    
    melhor_x, melhor_adequacao, historico = algoritmo_genetico(
        params["tamanho_populacao"],
        num_bits,
        x_min,
        x_max,
        params["num_geracoes"],
        taxa_crossover,
        taxa_mutacao,
        tamanho_torneio
    )
    
    resultados.append({
        'params': params,
        'melhor_x': melhor_x,
        'melhor_adequacao': melhor_adequacao
    })
    
    print(f"\n>>> RESULTADO FINAL:")
    print(f"    Melhor x encontrado: {melhor_x:.6f}")
    print(f"    f(x) máximo: {melhor_adequacao:.6f}")
    print("="*70)

# ========== RESUMO DOS RESULTADOS ==========
print("\n" + "="*70)
print("RESUMO COMPARATIVO DE TODOS OS TESTES")
print("="*70)
print(f"{'Pop':<5} {'Ger':<5} {'Melhor x':<15} {'f(x) Máximo':<15}")
print("-"*70)
for r in resultados:
    print(f"{r['params']['tamanho_populacao']:<5} "
          f"{r['params']['num_geracoes']:<5} "
          f"{r['melhor_x']:<15.6f} "
          f"{r['melhor_adequacao']:<15.6f}")

# Encontrar o melhor resultado geral
melhor_resultado = max(resultados, key=lambda x: x['melhor_adequacao'])
print("\n" + "="*70)
print(">>> MELHOR CONFIGURAÇÃO ENCONTRADA:")
print(f"    Parâmetros: {melhor_resultado['params']}")
print(f"    x ótimo: {melhor_resultado['melhor_x']:.6f}")
print(f"    f(x) máximo: {melhor_resultado['melhor_adequacao']:.6f}")
print("="*70)

# Valor teórico do máximo (para comparação)
print("\n[INFO] Valor máximo teórico:")
print("       Para x = 10: f(10) = 10² - 3(10) + 4 = 74.0")
print("       (máximo no intervalo [-10, 10])")