# 1. 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# 因果发现相关库
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

# 统计检验
from scipy import stats

# 2. 数据加载和预处理函数
def load_and_preprocess_data():
    """
    加载Auto MPG数据集并进行预处理
    """
    print("正在加载Auto MPG数据集...")
    
    try:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
        column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
        df = pd.read_csv(url, names=column_names, na_values='?', delim_whitespace=True)
        
        print(f"原始数据形状: {df.shape}")
        print(f"缺失值统计:\n{df.isnull().sum()}")
        
        # 移除非数值列
        df = df.drop('car_name', axis=1)
        
        # 处理缺失值
        print(f"删除缺失值前的数据形状: {df.shape}")
        df = df.dropna()
        print(f"删除缺失值后的数据形状: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("使用模拟数据代替...")
        
        # 生成模拟数据
        np.random.seed(42)
        n_samples = 300
        
        # 生成相关的汽车数据
        cylinders = np.random.choice([4, 6, 8], n_samples, p=[0.4, 0.4, 0.2])
        displacement = cylinders * 30 + np.random.normal(0, 20, n_samples)
        displacement = np.clip(displacement, 70, 500)
        
        horsepower = 0.3 * displacement + np.random.normal(0, 15, n_samples)
        horsepower = np.clip(horsepower, 40, 250)
        
        weight = 0.8 * displacement + 0.5 * horsepower + np.random.normal(0, 200, n_samples)
        weight = np.clip(weight, 1500, 5500)
        
        mpg = 50 - 0.01 * weight - 0.05 * horsepower + np.random.normal(0, 3, n_samples)
        mpg = np.clip(mpg, 8, 50)
        
        acceleration = 20 - 0.002 * weight + np.random.normal(0, 2, n_samples)
        acceleration = np.clip(acceleration, 8, 25)
        
        model_year = np.random.randint(70, 83, n_samples)
        origin = np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.2, 0.2])
        
        df = pd.DataFrame({
            'mpg': mpg,
            'cylinders': cylinders,
            'displacement': displacement,
            'horsepower': horsepower,
            'weight': weight,
            'acceleration': acceleration,
            'model_year': model_year,
            'origin': origin
        })
        
        print(f"生成的模拟数据形状: {df.shape}")
        
        return df

def explore_data(df):
    """
    数据探索和可视化
    """
    print("\n=== 数据探索 ===")
    print("数据基本信息:")
    print(df.info())
    print("\n描述性统计:")
    print(df.describe())
    
    # 相关性矩阵热图
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('变量相关性矩阵')
    
    # 分布图
    plt.subplot(2, 2, 2)
    df['mpg'].hist(bins=20, alpha=0.7)
    plt.title('MPG分布')
    plt.xlabel('Miles per Gallon')
    
    # 散点图
    plt.subplot(2, 2, 3)
    plt.scatter(df['weight'], df['mpg'], alpha=0.6)
    plt.xlabel('Weight')
    plt.ylabel('MPG')
    plt.title('Weight vs MPG')
    
    # 箱线图
    plt.subplot(2, 2, 4)
    df.boxplot(column='mpg', by='cylinders', ax=plt.gca())
    plt.title('MPG by Cylinders')
    
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

# 加载和探索数据
df = load_and_preprocess_data()
correlation_matrix = explore_data(df)

# 数据标准化（可选，某些算法可能需要）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)
data_original = df.to_numpy()

# 3. 因果发现算法实现
def run_causal_discovery_algorithms(data, node_names, alpha=0.05):
    """
    运行多种因果发现算法并比较结果
    """
    results = {}
    
    print(f"\n=== 运行因果发现算法 (alpha={alpha}) ===")
    
    # 1. PC算法 (约束型)
    print("运行PC算法...")
    try:
        pc_result = pc(data, alpha=alpha, indep_test='fisherz')
        results['PC'] = {
            'graph': pc_result.G,
            'graph_matrix': pc_result.G.graph,
            'algorithm_type': 'Constraint-based',
            'description': 'PC算法：基于条件独立性测试的约束型算法'
        }
        print("PC算法完成")
    except Exception as e:
        print(f"PC算法失败: {e}")
        results['PC'] = None
    
    # 2. FCI算法 (约束型，可处理隐变量)
    print("运行FCI算法...")
    try:
        fci_result = fci(data, alpha=alpha, indep_test='fisherz')
        results['FCI'] = {
            'graph': fci_result.G,
            'graph_matrix': fci_result.G.graph,
            'algorithm_type': 'Constraint-based (with latent variables)',
            'description': 'FCI算法：可处理隐变量的约束型算法'
        }
        print("FCI算法完成")
    except Exception as e:
        print(f"FCI算法失败: {e}")
        results['FCI'] = None
    
    # 3. GES算法 (评分型)
    print("运行GES算法...")
    try:
        # GES需要不同的参数格式
        ges_result = ges(data, score_func='local_score_BIC')
        results['GES'] = {
            'graph': ges_result.G,
            'graph_matrix': ges_result.G.graph,
            'algorithm_type': 'Score-based',
            'description': 'GES算法：基于贝叶斯信息准则的评分型算法'
        }
        print("GES算法完成")
    except Exception as e:
        print(f"GES算法失败: {e}")
        results['GES'] = None
    
    return results

def compare_algorithms(results, node_names):
    """
    比较不同算法的结果
    """
    print("\n=== 算法比较 ===")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) == 0:
        print("没有成功运行的算法")
        return
    
    print(f"成功运行的算法: {list(valid_results.keys())}")
    
    # 计算边的数量
    for name, result in valid_results.items():
        if result and 'graph_matrix' in result:
            graph_matrix = result['graph_matrix']
            num_edges = np.sum(graph_matrix != 0)
            print(f"{name}: {num_edges} 条边")
    
    # 计算算法间的一致性
    if len(valid_results) >= 2:
        algorithms = list(valid_results.keys())
        print("\n算法间边的重叠率:")
        
        for i in range(len(algorithms)):
            for j in range(i+1, len(algorithms)):
                alg1, alg2 = algorithms[i], algorithms[j]
                
                if (valid_results[alg1] and valid_results[alg2] and 
                    'graph_matrix' in valid_results[alg1] and 
                    'graph_matrix' in valid_results[alg2]):
                    
                    matrix1 = valid_results[alg1]['graph_matrix']
                    matrix2 = valid_results[alg2]['graph_matrix']
                    
                    # 计算边的重叠
                    edges1 = set(zip(*np.where(matrix1 != 0)))
                    edges2 = set(zip(*np.where(matrix2 != 0)))
                    
                    if len(edges1) > 0 and len(edges2) > 0:
                        overlap = len(edges1.intersection(edges2))
                        total_unique = len(edges1.union(edges2))
                        jaccard = overlap / total_unique if total_unique > 0 else 0
                        print(f"{alg1} vs {alg2}: Jaccard相似度 = {jaccard:.3f}")
    
    return valid_results

# 运行算法
results = run_causal_discovery_algorithms(data_original, df.columns.tolist())
valid_results = compare_algorithms(results, df.columns.tolist())

# 4. 高级可视化功能
def create_enhanced_graph_visualization(results, node_names, correlation_matrix=None):
    """
    创建增强的因果图可视化
    """
    if not results:
        print("没有可用的结果进行可视化")
        return
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_algorithms = len(valid_results)
    
    if n_algorithms == 0:
        print("没有成功的算法结果")
        return
    
    # 计算子图布局
    cols = min(3, n_algorithms)
    rows = (n_algorithms + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    if n_algorithms == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if n_algorithms == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, (alg_name, result) in enumerate(valid_results.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx] if n_algorithms > 1 else axes[0]
        
        # 获取图矩阵
        graph_matrix = result['graph_matrix']
        
        # 创建NetworkX图
        G = nx.DiGraph()
        
        # 添加节点
        for i, name in enumerate(node_names):
            G.add_node(name)
        
        # 添加边
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if graph_matrix[i, j] != 0:
                    # 边的权重表示关系强度
                    weight = abs(graph_matrix[i, j])
                    G.add_edge(node_names[i], node_names[j], weight=weight)
        
        # 布局
        try:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except:
            pos = nx.random_layout(G, seed=42)
        
        # 绘制图
        plt.sca(ax)
        
        # 节点颜色基于度数
        node_degrees = dict(G.degree())
        node_colors = [node_degrees.get(node, 0) for node in G.nodes()]
        
        # 边的粗细基于权重
        edges = G.edges()
        if len(edges) > 0:
            edge_weights = [G[u][v].get('weight', 1) for u, v in edges]
            edge_widths = [w * 2 for w in edge_weights]
        else:
            edge_widths = []
        
        # 绘制网络
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=1000, cmap='viridis', alpha=0.8, ax=ax)
        
        if len(edges) > 0:
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                                  edge_color='gray', arrows=True, 
                                  arrowsize=20, arrowstyle='->', ax=ax)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        # 标题
        n_edges = len(edges)
        ax.set_title(f'{alg_name}\n({result["algorithm_type"]})\n{n_edges} edges', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(n_algorithms, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('因果图发现结果比较', fontsize=16, y=1.02)
    plt.show()

def plot_adjacency_matrices(results, node_names):
    """
    绘制邻接矩阵热图
    """
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_algorithms = len(valid_results)
    
    if n_algorithms == 0:
        return
    
    fig, axes = plt.subplots(1, n_algorithms, figsize=(6*n_algorithms, 5))
    if n_algorithms == 1:
        axes = [axes]
    
    for idx, (alg_name, result) in enumerate(valid_results.items()):
        ax = axes[idx] if n_algorithms > 1 else axes[0]
        
        matrix = result['graph_matrix']
        
        # 创建热图
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # 设置标签
        ax.set_xticks(range(len(node_names)))
        ax.set_yticks(range(len(node_names)))
        ax.set_xticklabels(node_names, rotation=45, ha='right')
        ax.set_yticklabels(node_names)
        
        # 添加数值标注
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}' if matrix[i, j] != 0 else '',
                             ha="center", va="center", color="white" if abs(matrix[i, j]) > 0.5 else "black",
                             fontsize=8)
        
        ax.set_title(f'{alg_name} 邻接矩阵')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.show()

def create_summary_visualization(results, node_names, correlation_matrix):
    """
    创建综合分析可视化
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 相关性矩阵
    ax = axes[0, 0]
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('变量相关性矩阵')
    
    # 2. 算法边数比较
    ax = axes[0, 1]
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        alg_names = list(valid_results.keys())
        edge_counts = []
        
        for name, result in valid_results.items():
            matrix = result['graph_matrix']
            edge_count = np.sum(matrix != 0)
            edge_counts.append(edge_count)
        
        bars = ax.bar(alg_names, edge_counts, color=['skyblue', 'lightcoral', 'lightgreen'][:len(alg_names)])
        ax.set_title('各算法发现的边数')
        ax.set_ylabel('边数')
        
        # 在柱子上添加数值
        for bar, count in zip(bars, edge_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')
    
    # 3. 节点度分布
    ax = axes[1, 0]
    if valid_results:
        # 使用第一个成功的算法结果
        first_result = list(valid_results.values())[0]
        matrix = first_result['graph_matrix']
        
        # 计算入度和出度
        in_degrees = np.sum(matrix != 0, axis=0)
        out_degrees = np.sum(matrix != 0, axis=1)
        
        x = range(len(node_names))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], in_degrees, width, label='入度', alpha=0.8)
        ax.bar([i + width/2 for i in x], out_degrees, width, label='出度', alpha=0.8)
        
        ax.set_xlabel('变量')
        ax.set_ylabel('度数')
        ax.set_title('节点度分布 (基于第一个算法)')
        ax.set_xticks(x)
        ax.set_xticklabels(node_names, rotation=45, ha='right')
        ax.legend()
    
    # 4. 算法一致性分析
    ax = axes[1, 1]
    if len(valid_results) >= 2:
        algorithms = list(valid_results.keys())
        n_algs = len(algorithms)
        similarity_matrix = np.zeros((n_algs, n_algs))
        
        for i in range(n_algs):
            for j in range(n_algs):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    matrix1 = valid_results[algorithms[i]]['graph_matrix']
                    matrix2 = valid_results[algorithms[j]]['graph_matrix']
                    
                    edges1 = set(zip(*np.where(matrix1 != 0)))
                    edges2 = set(zip(*np.where(matrix2 != 0)))
                    
                    if len(edges1) > 0 and len(edges2) > 0:
                        overlap = len(edges1.intersection(edges2))
                        total_unique = len(edges1.union(edges2))
                        jaccard = overlap / total_unique if total_unique > 0 else 0
                        similarity_matrix[i, j] = jaccard
        
        im = ax.imshow(similarity_matrix, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(n_algs))
        ax.set_yticks(range(n_algs))
        ax.set_xticklabels(algorithms)
        ax.set_yticklabels(algorithms)
        
        for i in range(n_algs):
            for j in range(n_algs):
                ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                       ha="center", va="center")
        
        ax.set_title('算法间Jaccard相似度')
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, '需要至少2个算法\n进行比较', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('算法一致性分析')
    
    plt.tight_layout()
    plt.show()

# 执行可视化
print("\n=== 生成可视化 ===")
create_enhanced_graph_visualization(valid_results, df.columns.tolist(), correlation_matrix)
plot_adjacency_matrices(valid_results, df.columns.tolist())
create_summary_visualization(valid_results, df.columns.tolist(), correlation_matrix)

# 5. 结果分析和解释
def analyze_causal_relationships(results, node_names, df):
    """
    分析和解释因果关系发现的结果
    """
    print("\n=== 因果关系分析 ===")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("没有可用的结果进行分析")
        return
    
    # 汇总所有算法发现的边
    all_edges = set()
    edge_support = {}  # 记录每条边被多少算法支持
    
    for alg_name, result in valid_results.items():
        matrix = result['graph_matrix']
        
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if matrix[i, j] != 0:
                    edge = (node_names[i], node_names[j])
                    all_edges.add(edge)
                    
                    if edge not in edge_support:
                        edge_support[edge] = []
                    edge_support[edge].append((alg_name, matrix[i, j]))
    
    print(f"总共发现 {len(all_edges)} 条不同的因果边")
    
    # 按支持度排序边
    print("\n因果边及其支持情况:")
    print("-" * 60)
    
    for edge in sorted(all_edges, key=lambda x: len(edge_support[x]), reverse=True):
        cause, effect = edge
        support_count = len(edge_support[edge])
        support_algorithms = [f"{alg}({strength:.2f})" for alg, strength in edge_support[edge]]
        
        print(f"{cause} → {effect}")
        print(f"  支持算法数: {support_count}/{len(valid_results)}")
        print(f"  支持算法: {', '.join(support_algorithms)}")
        
        # 计算领域知识一致性
        domain_interpretation = interpret_relationship(cause, effect)
        print(f"  领域解释: {domain_interpretation}")
        print()
    
    # 生成因果路径分析
    analyze_causal_paths(valid_results, node_names)
    
    # 分析关键变量
    analyze_key_variables(valid_results, node_names, df)

def interpret_relationship(cause, effect):
    """
    基于汽车领域知识解释因果关系
    """
    relationships = {
        ('cylinders', 'displacement'): "汽缸数影响排量 - 符合机械原理",
        ('cylinders', 'horsepower'): "汽缸数影响马力 - 符合发动机原理", 
        ('displacement', 'horsepower'): "排量影响马力 - 发动机基本原理",
        ('displacement', 'weight'): "排量影响重量 - 大发动机更重",
        ('horsepower', 'mpg'): "马力影响油耗 - 高马力通常高油耗",
        ('weight', 'mpg'): "重量影响油耗 - 重车油耗高",
        ('weight', 'acceleration'): "重量影响加速 - 重车加速慢",
        ('horsepower', 'acceleration'): "马力影响加速 - 高马力加速快",
        ('model_year', 'mpg'): "年份影响油耗 - 技术进步提高效率",
        ('origin', 'mpg'): "产地影响油耗 - 不同国家设计理念",
    }
    
    return relationships.get((cause, effect), "需要进一步验证的关系")

def analyze_causal_paths(results, node_names):
    """
    分析因果路径
    """
    print("\n=== 因果路径分析 ===")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        return
    
    # 使用第一个算法的结果进行路径分析
    first_alg = list(valid_results.keys())[0]
    matrix = valid_results[first_alg]['graph_matrix']
    
    # 创建图
    G = nx.DiGraph()
    for i, name in enumerate(node_names):
        G.add_node(name)
    
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if matrix[i, j] != 0:
                G.add_edge(node_names[i], node_names[j], weight=abs(matrix[i, j]))
    
    # 寻找到MPG的所有路径
    if 'mpg' in node_names:
        print("影响MPG(燃油效率)的因果路径:")
        
        for source in node_names:
            if source != 'mpg' and nx.has_path(G, source, 'mpg'):
                try:
                    paths = list(nx.all_simple_paths(G, source, 'mpg', cutoff=3))
                    if paths:
                        shortest_path = min(paths, key=len)
                        print(f"  {source} → MPG: {' → '.join(shortest_path)}")
                except:
                    continue

def analyze_key_variables(results, node_names, df):
    """
    分析关键变量（中心性分析）
    """
    print("\n=== 关键变量分析 ===")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        return
    
    # 汇总所有算法的度分析
    centrality_scores = {name: {'in_degree': 0, 'out_degree': 0, 'total_degree': 0} for name in node_names}
    
    for alg_name, result in valid_results.items():
        matrix = result['graph_matrix']
        
        # 计算入度和出度
        in_degrees = np.sum(matrix != 0, axis=0)
        out_degrees = np.sum(matrix != 0, axis=1)
        
        for i, name in enumerate(node_names):
            centrality_scores[name]['in_degree'] += in_degrees[i]
            centrality_scores[name]['out_degree'] += out_degrees[i]
            centrality_scores[name]['total_degree'] += in_degrees[i] + out_degrees[i]
    
    # 输出分析结果
    print("变量中心性分析（汇总所有算法）:")
    print("-" * 40)
    
    # 按总度数排序
    sorted_vars = sorted(centrality_scores.items(), key=lambda x: x[1]['total_degree'], reverse=True)
    
    for name, scores in sorted_vars:
        print(f"{name}:")
        print(f"  入度: {scores['in_degree']:.1f} (被其他变量影响)")
        print(f"  出度: {scores['out_degree']:.1f} (影响其他变量)")
        print(f"  总度: {scores['total_degree']:.1f}")
        print()
    
    # 识别关键角色
    print("变量角色识别:")
    
    max_in = max(centrality_scores.values(), key=lambda x: x['in_degree'])
    max_out = max(centrality_scores.values(), key=lambda x: x['out_degree'])
    
    for name, scores in centrality_scores.items():
        if scores['in_degree'] == max_in['in_degree'] and scores['in_degree'] > 0:
            print(f"  {name}: 主要结果变量 (被多个变量影响)")
        elif scores['out_degree'] == max_out['out_degree'] and scores['out_degree'] > 0:
            print(f"  {name}: 主要原因变量 (影响多个变量)")
        elif scores['total_degree'] >= np.mean([s['total_degree'] for s in centrality_scores.values()]):
            print(f"  {name}: 中介变量 (连接性强)")

# 执行结果分析
analyze_causal_relationships(valid_results, df.columns.tolist(), df)

# 6. 模型验证和统计测试
def validate_causal_model(results, df, significance_level=0.05):
    """
    验证因果模型的统计显著性和稳定性
    """
    print("\n=== 模型验证 ===")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("没有可用的结果进行验证")
        return
    
    node_names = df.columns.tolist()
    
    # 1. 自举法稳定性测试
    print("1. 自举法稳定性测试:")
    bootstrap_stability_test(df, valid_results, node_names)
    
    # 2. 独立性测试验证
    print("\n2. 条件独立性测试验证:")
    independence_test_validation(df, valid_results, node_names, significance_level)
    
    # 3. 预测性能评估
    print("\n3. 预测性能评估:")
    prediction_performance_test(df, valid_results, node_names)

def bootstrap_stability_test(df, results, node_names, n_bootstrap=20):
    """
    使用自举法测试因果发现的稳定性
    """
    print(f"使用 {n_bootstrap} 次自举采样测试稳定性...")
    
    # 选择第一个成功的算法进行测试
    first_alg = list(results.keys())[0]
    original_matrix = results[first_alg]['graph_matrix']
    
    stable_edges = np.zeros_like(original_matrix)
    
    for i in range(n_bootstrap):
        # 自举采样
        bootstrap_indices = np.random.choice(len(df), len(df), replace=True)
        bootstrap_data = df.iloc[bootstrap_indices].to_numpy()
        
        try:
            # 重新运行算法
            if first_alg == 'PC':
                bootstrap_result = pc(bootstrap_data, alpha=0.05, indep_test='fisherz')
            elif first_alg == 'FCI':
                bootstrap_result = fci(bootstrap_data, alpha=0.05, indep_test='fisherz')
            elif first_alg == 'GES':
                bootstrap_result = ges(bootstrap_data, score_func='local_score_BIC')
            else:
                continue
                
            bootstrap_matrix = bootstrap_result.G.graph
            
            # 记录稳定的边
            stable_edges += (bootstrap_matrix != 0).astype(int)
            
        except Exception as e:
            print(f"自举采样 {i+1} 失败: {e}")
            continue
    
    # 计算稳定性得分
    stability_scores = stable_edges / n_bootstrap
    
    print(f"边稳定性分析 (稳定性阈值 > 0.5):")
    stable_edge_count = 0
    
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if original_matrix[i, j] != 0:
                stability = stability_scores[i, j]
                status = "稳定" if stability > 0.5 else "不稳定"
                print(f"  {node_names[i]} → {node_names[j]}: {stability:.2f} ({status})")
                if stability > 0.5:
                    stable_edge_count += 1
    
    total_edges = np.sum(original_matrix != 0)
    print(f"稳定边比例: {stable_edge_count}/{total_edges} ({stable_edge_count/total_edges*100:.1f}%)")

def independence_test_validation(df, results, node_names, alpha):
    """
    验证发现的因果关系是否违反条件独立性假设
    """
    print("检验因果边的条件独立性...")
    
    data = df.to_numpy()
    
    # 收集所有算法发现的边
    all_edges = set()
    for result in results.values():
        matrix = result['graph_matrix']
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if matrix[i, j] != 0:
                    all_edges.add((i, j))
    
    violations = 0
    total_tests = 0
    
    for i, j in all_edges:
        # 测试 X_i 和 X_j 在给定其他变量的条件下是否独立
        conditioning_set = [k for k in range(len(node_names)) if k != i and k != j]
        
        if len(conditioning_set) > 0:
            try:
                # 执行偏相关测试
                from scipy.stats import pearsonr
                
                # 简化的条件独立性测试（偏相关）
                if len(conditioning_set) <= 3:  # 限制条件集大小以避免维度诅咒
                    # 计算偏相关系数
                    partial_corr = calculate_partial_correlation(data, i, j, conditioning_set[:3])
                    
                    # 转换为统计量
                    n = len(data)
                    t_stat = partial_corr * np.sqrt((n - len(conditioning_set) - 2) / (1 - partial_corr**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - len(conditioning_set) - 2))
                    
                    total_tests += 1
                    if p_value > alpha:  # 应该拒绝独立性假设，但没有
                        violations += 1
                        print(f"  警告: {node_names[i]} → {node_names[j]} 可能不显著 (p={p_value:.3f})")
                
            except Exception as e:
                continue
    
    if total_tests > 0:
        violation_rate = violations / total_tests
        print(f"条件独立性测试结果: {violations}/{total_tests} 边可能不显著 ({violation_rate*100:.1f}%)")
    else:
        print("无法执行条件独立性测试")

def calculate_partial_correlation(data, i, j, conditioning_set):
    """
    计算偏相关系数
    """
    try:
        # 简化实现：使用线性回归残差
        from sklearn.linear_model import LinearRegression
        
        X_cond = data[:, conditioning_set]
        y_i = data[:, i]
        y_j = data[:, j]
        
        # 回归 X_i 对条件集
        reg_i = LinearRegression().fit(X_cond, y_i)
        residual_i = y_i - reg_i.predict(X_cond)
        
        # 回归 X_j 对条件集
        reg_j = LinearRegression().fit(X_cond, y_j)
        residual_j = y_j - reg_j.predict(X_cond)
        
        # 计算残差间的相关性
        correlation, _ = pearsonr(residual_i, residual_j)
        return correlation
        
    except Exception:
        return 0.0

def prediction_performance_test(df, results, node_names):
    """
    测试因果模型的预测性能
    """
    print("评估因果模型的预测能力...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    # 选择MPG作为目标变量
    if 'mpg' not in node_names:
        print("数据中没有MPG变量，跳过预测测试")
        return
    
    target_idx = node_names.index('mpg')
    
    # 使用第一个算法的结果
    if not results:
        return
    
    first_result = list(results.values())[0]
    causal_matrix = first_result['graph_matrix']
    
    # 找到影响MPG的变量
    causal_predictors = []
    for i in range(len(node_names)):
        if causal_matrix[i, target_idx] != 0:
            causal_predictors.append(i)
    
    if len(causal_predictors) == 0:
        print("没有发现影响MPG的因果变量")
        return
    
    # 准备数据
    X_causal = df.iloc[:, causal_predictors].to_numpy()
    X_all = df.drop('mpg', axis=1).to_numpy()
    y = df['mpg'].to_numpy()
    
    # 分割数据
    X_causal_train, X_causal_test, X_all_train, X_all_test, y_train, y_test = train_test_split(
        X_causal, X_all, y, test_size=0.3, random_state=42
    )
    
    # 训练模型
    causal_model = LinearRegression().fit(X_causal_train, y_train)
    full_model = LinearRegression().fit(X_all_train, y_train)
    
    # 预测
    y_pred_causal = causal_model.predict(X_causal_test)
    y_pred_full = full_model.predict(X_all_test)
    
    # 评估
    mse_causal = mean_squared_error(y_test, y_pred_causal)
    mse_full = mean_squared_error(y_test, y_pred_full)
    r2_causal = r2_score(y_test, y_pred_causal)
    r2_full = r2_score(y_test, y_pred_full)
    
    print(f"因果模型预测性能:")
    print(f"  使用变量: {[node_names[i] for i in causal_predictors]}")
    print(f"  MSE: {mse_causal:.3f}")
    print(f"  R²: {r2_causal:.3f}")
    
    print(f"完整模型预测性能:")
    print(f"  使用所有变量 ({len(node_names)-1} 个)")
    print(f"  MSE: {mse_full:.3f}")
    print(f"  R²: {r2_full:.3f}")
    
    efficiency = len(causal_predictors) / (len(node_names) - 1)
    performance_ratio = r2_causal / r2_full if r2_full > 0 else 0
    
    print(f"模型效率: {efficiency:.2f} (使用 {efficiency*100:.1f}% 的变量)")
    print(f"性能比率: {performance_ratio:.2f} (因果模型/完整模型)")

# 执行模型验证
validate_causal_model(valid_results, df)

# 7. 总结和建议
def generate_summary_report(results, df):
    """
    生成分析总结报告
    """
    print("\n" + "="*60)
    print("           因果图发现分析总结报告")
    print("="*60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    print(f"数据集信息:")
    print(f"  - 样本数量: {len(df)}")
    print(f"  - 变量数量: {len(df.columns)}")
    print(f"  - 变量列表: {', '.join(df.columns.tolist())}")
    
    print(f"\n算法执行情况:")
    print(f"  - 成功运行的算法: {len(valid_results)}")
    print(f"  - 算法列表: {', '.join(valid_results.keys())}")
    
    if valid_results:
        total_edges = sum(np.sum(result['graph_matrix'] != 0) for result in valid_results.values())
        avg_edges = total_edges / len(valid_results)
        print(f"  - 平均发现边数: {avg_edges:.1f}")
        
        print(f"\n关键发现:")
        
        # 找到被多个算法支持的边
        edge_support = {}
        for alg_name, result in valid_results.items():
            matrix = result['graph_matrix']
            node_names = df.columns.tolist()
            
            for i in range(len(node_names)):
                for j in range(len(node_names)):
                    if matrix[i, j] != 0:
                        edge = (node_names[i], node_names[j])
                        if edge not in edge_support:
                            edge_support[edge] = 0
                        edge_support[edge] += 1
        
        # 高度一致的边
        consensus_edges = [(edge, count) for edge, count in edge_support.items() 
                          if count >= len(valid_results) * 0.6]
        
        if consensus_edges:
            print(f"  - 高度一致的因果关系 (≥60%算法支持):")
            for (cause, effect), count in sorted(consensus_edges, key=lambda x: x[1], reverse=True):
                print(f"    • {cause} → {effect} (支持度: {count}/{len(valid_results)})")
        
        print(f"\n建议:")
        print(f"  1. 重点关注高支持度的因果关系")
        print(f"  2. 对低稳定性的边进行进一步验证")
        print(f"  3. 考虑收集更多数据以提高算法稳定性")
        print(f"  4. 结合领域知识验证因果发现结果")
        
        if len(valid_results) >= 2:
            print(f"  5. 多算法一致的结果更可信")
        else:
            print(f"  5. 建议运行多个算法进行交叉验证")
    
    print("\n" + "="*60)
    print("分析完成！请查看上述可视化结果和统计分析。")
    print("="*60)

# 生成最终报告
generate_summary_report(valid_results, df)

print(f"""
使用说明和扩展建议:

1. 运行环境要求:
   - Python 3.7+
   - 必需库: causal-learn, pandas, numpy, matplotlib, seaborn, networkx, scikit-learn, scipy

2. 主要功能:
   - 自动数据加载和预处理 (支持在线数据和模拟数据)
   - 多种因果发现算法比较 (PC, FCI, GES)
   - 丰富的可视化分析
   - 结果解释和领域知识验证
   - 模型稳定性和统计显著性测试

3. 可以尝试的修改:
   - 调整alpha参数 (0.01, 0.05, 0.1) 观察结果变化
   - 尝试不同的独立性测试方法
   - 添加背景知识约束
   - 使用更大的数据集
   - 探索非线性因果关系

4. 进一步扩展:
   - 集成更多因果发现算法 (LINGAM, NOTEARS等)
   - 添加因果效应估计功能
   - 支持时间序列因果分析
   - 实现因果图的交互式可视化
   - 添加因果推断的反事实分析
""")

# 如果需要保存结果
def save_results(results, filename_prefix="causal_analysis"):
    """
    保存分析结果到文件
    """
    import pickle
    import json
    
    # 保存结果到pickle文件
    with open(f"{filename_prefix}_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # 保存简化的结果到JSON文件
    simplified_results = {}
    for alg_name, result in results.items():
        if result is not None:
            simplified_results[alg_name] = {
                'algorithm_type': result['algorithm_type'],
                'description': result['description'],
                'num_edges': int(np.sum(result['graph_matrix'] != 0))
            }
    
    with open(f"{filename_prefix}_summary.json", 'w', encoding='utf-8') as f:
        json.dump(simplified_results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到 {filename_prefix}_results.pkl 和 {filename_prefix}_summary.json")

# 取消注释下面这行来保存结果
# save_results(valid_results)
