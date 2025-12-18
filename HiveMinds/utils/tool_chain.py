import networkx as nx
from typing import List, Tuple
from typing import Union


class Tool_Chain_Analyzer:
    """Tool chain analyzer class
    1-based index
    """
    
    def __init__(self, execute_tools=[]):
        """Initialize the directed graph"""
        self.tool_chain = nx.DiGraph()
        self.execute_tools = execute_tools      # tool_execute_order
        
    def clear(self):
        """clear the graph"""
        self.tool_chain.clear()
        self.execute_tools = []
    
    # step_num -> tool_name
    def step_to_tool_name(self, step_num):
        if len(self.execute_tools) >= step_num:
            return self.execute_tools[step_num-1]
        else:
            raise ValueError(f"节点 '{step_num}' 不在可执行工具列表中")
        
    # tool_name -> step_num
    def tool_name_to_step(self, tool_name):
        if tool_name in self.execute_tools:
            return self.execute_tools.index(tool_name)+1
        else:
            raise ValueError(f"工具 '{tool_name}' 不在可执行工具列表中")
        
    def add_tool_node(self, node):
        """Add a node to the graph"""
        if isinstance(node, int):
            if len(self.execute_tools) >= node:
                node = self.execute_tools[node-1]
            else:
                raise ValueError(f"节点 '{node}' 不在可执行工具列表中")
        
        self.tool_chain.add_node(node)
        self.execute_tools.append(node)
        
    def add_tool_nodes_from(self, nodes: list):
        """Add multiple nodes to the graph"""
        try:
            nodes = [self.execute_tools[node-1] if isinstance(node, int) else node for node in nodes]
        except Exception as e:
            print(e)
            raise ValueError(f"存在节点不在可执行工具列表中")
        
        self.tool_chain.add_nodes_from(nodes)
        self.execute_tools.extend(nodes)
    
    def add_edge(self, source, target, tool_name_list=None):
        """Add a directed edge to the graph"""
        if isinstance(source, int):
            source = self.step_to_tool_name(source)
        if isinstance(target, int):
            target = self.step_to_tool_name(target)
            
        # tool_names = tool_name_list if tool_name_list else self.execute_tools
        # source, target = tool_names[source-1], tool_names[target-1]
        self.tool_chain.add_edge(source, target)
    
    def add_edges_from(self, edges: list, tool_name_list=None):
        """Add multiple directed edges to the graph"""
        # tool_names = tool_name_list if tool_name_list else self.execute_tools
        # tool_edges = [(tool_names[edge[0]-1], tool_names[edge[1]-1]) for edge in edges]
        
        tool_edges = [(self.step_to_tool_name(edge[0]), 
                       self.step_to_tool_name(edge[1])) 
                      if isinstance(edge[0], int) and isinstance(edge[1], int) else edge 
                      for edge in edges ]
        self.tool_chain.add_edges_from(tool_edges)
        
    def add_current_connection(self, tool_name, conn_action_step: set, tool_name_list=None):
        """Add the current tool's connection steps
            conn_action_step: set of connection steps, int elements
        """
        current_step = self.tool_name_to_step(tool_name)
        edges = [(step, current_step) for step in conn_action_step]
        self.add_edges_from(edges)
    
    def get_next_nodes(self, node):
        """Get the direct successors of the specified node"""
        if node not in self.tool_chain:
            raise ValueError(f"节点 '{node}' 不在图中")
        return list(self.tool_chain.successors(node))
    
    def get_prev_nodes(self, node):
        """Get the direct predecessors of the specified node"""
        if node not in self.tool_chain:
            raise ValueError(f"节点 '{node}' 不在图中")
        return list(self.tool_chain.predecessors(node))
    
    def get_all_successors(self, node, include_self=True):
        """
        Get all the successors of the specified node, including direct and indirect ones
        include direct and indirect successors
        
        Parameters:
            include_self: whether to include the starting node itself
        """
        if node not in self.tool_chain:
            raise ValueError(f"节点 '{node}' 不在图中")
        
        successors = set(nx.descendants(self.tool_chain, node))
        if include_self:
            successors.add(node)
        return list(successors)
    
    def get_common_successors(self, nodes, include_self=True):
        """
        Get the common successors of multiple nodes, including direct and indirect ones
        
        Parameters:
            nodes: a list of nodes
            include_self: whether to include the starting nodes themselves
        """
        if not nodes:
            return []
            
        common = set(self.get_all_successors(nodes[0], include_self))
        for node in nodes[1:]:
            common &= set(self.get_all_successors(node, include_self))
        return list(common)
    
    def topological_sort_successors(self, nodes, include_self=True, remove_internal_edges=False):
        """
        Topological sort of successors of multiple nodes, including direct and indirect ones

        """
        # 收集所有后续节点（包括起始节点）
        nodes = nodes if isinstance(nodes, list) else [nodes]
        all_nodes = set()
        for node in nodes:
            all_nodes.update(self.get_all_successors(node, include_self))
        
        # 如果没有节点，返回空列表
        if not all_nodes:
            return []
        
        # 构建子图
        subgraph = self.tool_chain.subgraph(all_nodes)
        
        try:
            # 进行拓扑排序
            # return list(nx.topological_sort(subgraph))
            sorted_nodes = list(nx.topological_sort(subgraph))
        
            # 选择性删除内部边
            if remove_internal_edges:
                edges_to_remove = list(subgraph.edges())
                self.tool_chain.remove_edges_from(edges_to_remove)
                print(f"已删除 {len(edges_to_remove)} 条内部边")
                
            return sorted_nodes
        except nx.NetworkXUnfeasible:
            # 处理环（简化处理：移除环中的边）
            edges = list(nx.find_cycle(subgraph))
            self.tool_chain.remove_edges_from(edges)
            print(f"警告：检测到环，已移除边 {edges} 以进行拓扑排序")
            return self.topological_sort_successors(nodes, include_self)
        
    def remove_edges_between_nodes(self, nodes, condition=None):
        """Remove edges between specified nodes, support condition filtering
        
        Parameters:
            nodes: 
            condition: a function that takes an edge (u, v) and returns True or False, 
        """
        if not nodes:
            return
        
        # 创建子图并获取所有内部边
        subgraph = self.tool_chain.subgraph(nodes)
        edges_to_remove = list(subgraph.edges())
        
        if condition:
            edges_to_remove = [edge for edge in edges_to_remove if condition(edge)]
        
        self.tool_chain.remove_edges_from(edges_to_remove)
        return edges_to_remove
    

    # def truncate(self, nodes: list | int | str, show=False):
    def truncate(self, nodes: Union[list, int, str], show=False):

        if isinstance(nodes, int):
            nodes = [self.step_to_tool_name(nodes)]
        elif isinstance(nodes, str):
            nodes = [nodes]
        else:
            nodes = [self.step_to_tool_name(node) 
                     if isinstance(node, int) else node for node in nodes]
        
        sorted_nodes = self.topological_sort_successors(nodes)
        if show:
            self.visualize(highlight_nodes=sorted_nodes)
        # edges_to_remove = self.remove_edges_between_nodes(sorted_nodes)
        edges_to_remove = self.remove_all_edges_of_nodes(sorted_nodes)
        return sorted_nodes, edges_to_remove
    
    def remove_all_edges_of_nodes(self, nodes):
        """
        Remove all input and output edges of the specified node and its subsequent nodes.
        """
        if not nodes:
            return []
        
        target_nodes = set(nodes)
        
        input_edges = [(u, v) for u, v in self.tool_chain.edges() if v in target_nodes]
        output_edges = [(u, v) for u, v in self.tool_chain.edges() if u in target_nodes]
        edges_to_remove = list(set(input_edges + output_edges))
        
        self.tool_chain.remove_edges_from(edges_to_remove)
        
        return edges_to_remove
    
    def visualize(self, highlight_nodes=None):
        try:
            import matplotlib.pyplot as plt
            
            # pos = nx.spring_layout(self.tool_chain)
            # pos = nx.nx_agraph.graphviz_layout(self.tool_chain, prog='dot', args='-Grankdir=LR')
            pos = nx.nx_agraph.graphviz_layout(self.tool_chain, prog='dot', args='-Grankdir=LR -Granksep=1.0 -Gnodesep=0.5')
            # pos = nx.shell_layout(self.tool_chain)
            # pos = nx.spectral_layout(self.tool_chain)
            # pos = nx.spiral_layout(self.tool_chain)
            # pos = nx.circular_layout(self.tool_chain)
            # pos = nx.random_layout(self.tool_chain)
            # pos = nx.kamada_kawai_layout(self.tool_chain)
            # pos = nx.planar_layout(self.tool_chain)
            # pos = nx.multipartite_layout(self.tool_chain, subset_key="layer")

            plt.figure(figsize=(12, 8))
            if highlight_nodes:
                # 高亮显示指定节点及其后续节点
                all_highlighted = set()
                for node in highlight_nodes:
                    all_highlighted.update(self.get_all_successors(node, True))
                
                node_colors = ['lightblue' if n not in all_highlighted else 'orange' 
                              for n in self.tool_chain.nodes()]
                
                sorted_nodes = self.topological_sort_successors(highlight_nodes, True)
                plt.title(f"节点 {highlight_nodes} 的拓扑排序: {sorted_nodes}")
            else:
                node_colors = ['lightblue'] * len(self.tool_chain.nodes())
                plt.title("有向图结构")
            
            
            # nx.draw(self.tool_chain, pos, with_labels=True, 
            #         node_color=node_colors, arrows=True)
            nx.draw(self.tool_chain, pos, with_labels=True, node_size=1200, node_color=node_colors,
                    font_size=10, arrows=True, arrowstyle="->", arrowsize=20)
            # nx.draw(self.tool_chain, pos, with_labels=True, node_size=1200, node_color="lightblue",
            #         font_size=10, arrows=True, arrowstyle="->", arrowsize=20)
            plt.show()
        except ImportError:
            print("安装 matplotlib 库可可视化图结构")


# 使用示例
if __name__ == "__main__":
    # 创建图并添加边
    tool_name_list = ['A', 'B', 'C', 'D', 'E', 'F']
    analyzer = Tool_Chain_Analyzer(tool_name_list)
    # analyzer.add_edges_from([
    #     ('A', 'B'), ('A', 'C'),
    #     ('B', 'D'), ('C', 'E'),
    #     ('D', 'F'), ('E', 'F')
    # ])
    
    edges = [
        (1, 2), (1, 3),
        (2, 4), (3, 5),
        (4, 6), (5, 6)
    ]
    analyzer.add_edges_from(edges, tool_name_list)
    
    
    # 打印后继节点
    # 查询节点A的所有后续节点（包括A本身）
    print("节点 'B' 的所有后续节点（包括B）:", analyzer.get_all_successors('B'))
    print("节点 'E' 的所有后续节点（包括E）:", analyzer.get_all_successors('E'))
    
    # 查询节点A和B的共同后续节点（包括A和B本身）
    print("节点 'E' 和 'B' 的共同后续节点:", analyzer.get_common_successors(['E', 'B']))
    
    # 对节点A和B的所有后续节点进行拓扑排序（包括A和B本身）
    sorted_nodes = analyzer.topological_sort_successors(['E', 'B'])
    print("节点 'E' 和 'B' 的拓扑排序:", sorted_nodes)
    
    # 可视化图（高亮显示节点A和B及其后续节点）
    analyzer.visualize(highlight_nodes=['E', 'B'])
    
    ########################################################################
    Tool_Names = [
        "define_pde",
        "define_reference_solution",
        "define_domain",
        "define_initial_condition",
        "define_boundary_condition",
        "create_training_data",
        "create_network",
        "train_model",
        "train_model_LBFGS",
        "visualize_and_save",
    ]
    tool_chain_analyzer = Tool_Chain_Analyzer(Tool_Names)
    tool_chain_analyzer.add_tool_nodes_from(Tool_Names)
    # tool_chain_analyzer.add_edges_from([
    #     ('define_pde', 'create_training_data'),
    #     ('define_reference_solution', 'create_training_data'),
    #     ('define_domain', 'define_initial_condition'),
    #     ('define_domain', 'define_boundary_condition'),
    #     ('define_domain', 'create_training_data'),
    #     ('define_boundary_condition', 'create_training_data'),
    #     ('define_initial_condition', 'create_training_data'),
    #     ('create_training_data', 'train_model'),
    #     ('create_network', 'train_model'),
    #     ('train_model', 'train_model_LBFGS'),
    #     ('train_model', 'visualize_and_save'),
    #     ('train_model_LBFGS', 'visualize_and_save'),
    # ])
    tool_chain_analyzer.add_edges_from([
        (1, 6), (2, 6), 
        (3, 4), (3, 5),
        (3, 6), (4, 6),
        (5, 6), (6, 8),
        (7, 8), (8, 9),
        (8, 10), (9, 10),
    ])

    
    # 查询节点A的所有后续节点（包括A本身）
    print("节点 'define_initial_condition' 的所有后续节点（包括define_initial_condition）:", tool_chain_analyzer.get_all_successors('define_initial_condition'))
    print("节点 'create_network' 的所有后续节点（包括 create_network）:", tool_chain_analyzer.get_all_successors('create_network'))

    # 查询节点A和B的共同后续节点（包括A和B本身）
    print("节点 'define_initial_condition' 和 'create_network' 的共同后续节点:", tool_chain_analyzer.get_common_successors(['define_initial_condition', 'create_network']))
    
    # 对节点A和B的所有后续节点进行拓扑排序（包括A和B本身）
    sorted_nodes = tool_chain_analyzer.topological_sort_successors(['define_initial_condition', 'create_network'])
    print("节点 'define_initial_condition' 和 'create_network' 的拓扑排序:", sorted_nodes)
    
    # 可视化图（高亮显示节点A和B及其后续节点）
    tool_chain_analyzer.visualize(highlight_nodes=['define_initial_condition', 'create_network'])
    
    
    