import deepxde as dde
import numpy as np
import torch
import requests
from openai import OpenAI
import os
import ast
import matplotlib.pyplot as plt
from deepxde.utils.external import _pack_data
from typing import List, Dict
import json

def make_safe_globals():
    return {
        "dde": dde,
        "torch": torch,
        "np": np,
        "__builtins__": {
            "range": range,
            "float": float,
            "int": int
        }
    }


def extract_program(response: str, last_only=True):
    """
    从字符串中提取代码 以```python开头, ```结尾
    """
    program = ""
    start = False
    for line in response.split("\n"):
        if line.startswith("```python"):
            if last_only:
                program = ""  # 只提取最后一个程序
            else:
                program += "\n# ========\n"
            start = True
        elif line.startswith("```"):
            start = False
        elif start:
            program += line + "\n"
    return program


# 
def save_token_file(checkpoint_path: str, new_data: Dict, save_interval: int = 10):
    
    try:
        # 读取现有数据（若文件不存在则初始化空列表）
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint: Dict = json.load(f)
        except FileNotFoundError:
            checkpoint = {}
        
        checkpoint[len(checkpoint)] = new_data
        
        # checkpoint = update_json_file(checkpoint, new_data)

        # 原子性写入：先写入临时文件，再替换原文件
        with open(f"{checkpoint_path}.tmp", "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=4, ensure_ascii=False)
        
        # 替换原文件（确保写入完整）
        # import os
        os.replace(f"{checkpoint_path}.tmp", checkpoint_path)
        # print(f"Checkpoint saved at {checkpoint_path} (Total entries: {len(checkpoint_list)})")
    
    except Exception as e:
        print(f"保存检查点失败: {e}")
        # 可添加重试逻辑或忽略错误继续训练


# 工具 1: 定义PDE
def define_pde(equation: str, request_llm):
    """
    定义偏微分方程。
    参数:
        equation: 偏微分方程的LaTex格式表达式
    返回:
        pde: 偏微分方程的函数
        pure_code: 提取出的纯Python代码
    """
    pde_def_prompt_path = os.path.join(os.path.dirname(__file__), "desc", "pde_def_prompt.txt")
    if not os.path.exists(pde_def_prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {pde_def_prompt_path}")
    with open(pde_def_prompt_path, "r", encoding="utf-8") as f:
        pde_def_prompt = f.read()

    pure_code = request_llm(system_prompt=pde_def_prompt, user_prompt=equation)
    try:
        tree = ast.parse(pure_code)
        if not any(isinstance(node, ast.FunctionDef) and node.name == "pde" for node in tree.body):
            raise ValueError(f"Generated code must define a 'pde' function: {pure_code}")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in generated code: {pure_code}\nError: {str(e)}")

    local_vars = {"dde": dde, "torch": torch}
    try:
        exec(compile(tree, "<ast>", "exec"), make_safe_globals(), local_vars)
    except Exception as e:
        raise ValueError(f"Runtime error in generated code: {pure_code}\nError: {str(e)}")
    if "pde" not in local_vars:
        raise ValueError(f"No 'pde' function defined after execution: {pure_code}")

    return local_vars["pde"], pure_code


# 工具 2: 定义解析解
def define_reference_solution(solution: str, request_llm):
    """
    定义参考解。
    参数:
        solution: 参考解的LaTex格式表达式
    返回:
        ref_func: 参考解的函数
        pure_code: 提取出的纯Python代码
    """
    # 使用相对路径加载提示文件
    ref_sol_def_prompt_path = os.path.join(os.path.dirname(__file__), "desc", "ref_sol_def_prompt.txt")
    if not os.path.exists(ref_sol_def_prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {ref_sol_def_prompt_path}")
    with open(ref_sol_def_prompt_path, "r", encoding="utf-8") as f:
        ref_sol_def_prompt = f.read()

    # 获取LLM生成的代码
    pure_code = request_llm(system_prompt=ref_sol_def_prompt, user_prompt=solution)

    # 解析AST并验证
    try:
        tree = ast.parse(pure_code)
        if not any(isinstance(node, ast.FunctionDef) and node.name == "ref_func" for node in tree.body):
            raise ValueError(f"Generated code must define a 'ref_func' function: {pure_code}")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in generated code: {pure_code}\nError: {str(e)}")

    # 执行代码
    local_vars = {"np": np}
    try:
        exec(compile(tree, "<ast>", "exec"), make_safe_globals(), local_vars)
    except Exception as e:
        raise ValueError(f"Runtime error in generated code: {pure_code}\nError: {str(e)}")
    if "ref_func" not in local_vars:
        raise ValueError(f"Extracted code did not define a 'ref_func' function: {pure_code}")

    return local_vars["ref_func"], pure_code


# 工具 3: 定义计算域
def define_domain(geom_type, geom_range, is_time_dependent=False, time_range=None):
    """
    定义空间和时间计算域。
    参数:
        geom_type: 几何类型
        geom_range: 几何范围
        is_time_dependent: 是否时间依赖
        time_range: 时间范围
    返回:
        geomtime: 空间-时间几何对象
    """
    Geom_dict = {
        "Interval": dde.geometry.Interval,
        "Rectangle": dde.geometry.Rectangle,
        "Polygon": dde.geometry.Polygon,
        "Disk": dde.geometry.Disk,
        "Ellipse": dde.geometry.Ellipse,
        "Triangle": dde.geometry.Triangle,
        "Cuboid": dde.geometry.Cuboid,
        "Sphere": dde.geometry.Sphere,
        "Hypercube": dde.geometry.Hypercube,
        "Hypersphere": dde.geometry.Hypersphere
    }

    geom_func = Geom_dict.get(geom_type)
    if geom_func is None:
        raise ValueError(f"Invalid geometry type:{geom_type}. Supported types: {list(Geom_dict.keys())}")


    if geom_type in ["Interval", "Rectangle",  "Cuboid", "Sphere", "Hypercube", "Hypersphere", "Disk"]:
        geom = geom_func(geom_range[0], geom_range[1])
    elif geom_type == "Polygon":
        geom = geom_func(geom_range)
    elif geom_type == "Ellipse":
        geom = geom_func(geom_range[0], geom_range[1], geom_range[2], geom_range[3])
    elif geom_type == "Triangle":
        geom = geom_func(geom_range[0], geom_range[1], geom_range[2])

    if is_time_dependent:
        if time_range is None or not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
            raise ValueError("time_range must be a list or tuple of length 2 (e.g., [tmin, tmax])")
        try:
            timedomain = dde.geometry.TimeDomain(time_range[0], time_range[1])
            return dde.geometry.timedomain.GeometryXTime(geom, timedomain)
        except Exception as e:
            raise ValueError(f"Failed to create time domain with range {time_range}: {str(e)}")

    return geom

# 工具 4：定义初始条件
def define_initial_condition(geomtime, ic_description: str, request_llm):
    """
        定义初始条件。
        参数:
            geomtime: 空间-时间几何对象
            ic_description: 初始条件描述, 这是一个字符串，是和初始条件相关的描述，用来和语言模型交互，从而生成初始条件。
        返回:
            initial_conditions: 初始条件对象列表
            pure_code: 提取出的定义初始条件的Python代码
    """
    ic_def_prompt_path = os.path.join(os.path.dirname(__file__), "desc", "ic_def_prompt.txt")
    if not os.path.exists(ic_def_prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {ic_def_prompt_path}")
    with open(ic_def_prompt_path, "r", encoding="utf-8")as f:
        ic_def_prompt = f.read()
    pure_code = request_llm(system_prompt=ic_def_prompt, user_prompt=ic_description)

    local_vars = {
        "dde": dde,
        "np": np,
        "torch": torch,
        "geomtime": geomtime
    }
    try:
        exec(pure_code, make_safe_globals(), local_vars)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in extracted code: {pure_code}\nError: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error executing extracted code: {pure_code}\nError: {str(e)}")

    initial_conditions = []
    i = 1
    while f"ic{i}" in local_vars:
        initial_conditions.append(local_vars[f"ic{i}"])
        i += 1

    if not initial_conditions:
        raise ValueError("No initial conditions found in the generated code.")

    return initial_conditions, pure_code


# 工具 5：定义边界条件
def define_boundary_condition(geomtime,  bc_description: str, request_llm):
    """
        定义边界条件。因为实际中边界条件的复杂性，需要和LLM的api交互来生成边界条件。
        参数:
            geomtime: 空间-时间几何对象
            bc_description: 边界条件描述, 这是一个字符串，是和边界条件相关的描述，用来和语言模型交互，从而生成边界条件。
        返回:
            boundary_conditions：边界条件对象列表
            pure_code: 提取出的定义边界条件的Python代码
    """
    bc_def_prompt_path = os.path.join(os.path.dirname(__file__), "desc", "bc_def_prompt.txt")
    if not os.path.exists(bc_def_prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {bc_def_prompt_path}")
    with open(bc_def_prompt_path, "r", encoding="utf-8")as f:
        bc_def_prompt = f.read()
    pure_code = request_llm(system_prompt=bc_def_prompt, user_prompt=bc_description)

    local_vars = {
        "dde": dde,
        "np": np,
        "geom": geomtime,
        "geomtime":geomtime,
        "len": len  # 添加 len 函数
    }
    try:
        exec(pure_code, make_safe_globals(), local_vars)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in extracted code: {pure_code}\nError: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error executing extracted code: {pure_code}\nError: {str(e)}")

    boundary_conditions = []
    i = 1
    while f"bc{i}" in local_vars:
        boundary_conditions.append(local_vars[f"bc{i}"])
        i += 1

    if not boundary_conditions:
        raise ValueError("No boundary conditions found in the generated code.")

    return boundary_conditions, pure_code


# 工具 6：创建训练数据
def create_training_data(geomtime, pde_func,
                         num_domain, num_boundary=100, num_initial=100, num_test=None,
                         bc=None, ic=None, ref_func=None, is_time_dependent=False):
    """
    创建PDE训练数据。
    参数:
        geomtime: 空间-时间几何对象
        pde_func: PDE函数
        num_domain, num_boundary, num_initial, num_test: 采样点数
        bc: 边界条件对象列表,包括所需的边界条件对象
        ic: 初始条件对象列表,包括所需的初始条件对象
        ref_func: 参考解函数
        is_time_dependent: 是否时间依赖
    返回:
        data: 训练数据对象
    """
    if is_time_dependent:
        conditions = bc + ic if bc else ic
        return dde.data.TimePDE(
            geomtime,
            pde_func,
            conditions,
            num_domain=num_domain,
            num_boundary=num_boundary,
            num_initial=num_initial,
            solution=ref_func,
            num_test=num_test
        )
    else:
        conditions = bc
        return dde.data.PDE(
            geomtime,
            pde_func,
            conditions,
            num_domain=num_domain,
            num_boundary=num_boundary,
            solution=ref_func,
            num_test=num_test
        )

# 工具7：创建神经网络
def create_network(input_dim, output_dim, hidden_layers, activation="tanh",
                   initializer="Glorot uniform"):
    """
    创建神经网络。
    参数:
        input_dim: 输入维度
        output_dim: 输出维度
        hidden_layers: 隐藏层神经元列表
        activation: 激活函数
        initializer: 初始化方法
    返回:
        net: 神经网络对象
    """
    layer_size = [input_dim] + hidden_layers + [output_dim]
    net = dde.nn.FNN(layer_size, activation, initializer)
    
    # TODO: save config
    with open(os.path.join(os.environ["OUTPUT_DIR"], "net_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "layer_sizes": layer_size,  # 网络层数：1输入 + 3×50隐藏层 + 1输出
            "activation": activation,
            "kernel_initializer": initializer,
        }, f,  ensure_ascii=False, indent=4)
    
    return net

# 工具8：训练模型
def train_model(data, net,
                optimizer="adam", lr=0.001,
                iterations=15000, metrics=None,
                resample=False, period=None):
    """
    编译并训练模型。
    参数:
        data: 训练数据对象
        net: 神经网络对象
        optimizer: 优化器
        lr: 学习率
        iterations: 训练迭代次数
        metrics: 评估指标
        resample: 是否重采样
        period: 重采样周期
    返回:
        model: 训练好的模型
        losshistory: 损失历史
        train_state: 训练状态
    """
    if type(data.geom).__name__ == "Polygon":
        net=net.double()
    if type(data.geom).__name__ == "GeometryXTime":
        if type(data.geom.geometry).__name__ == "Polygon":
            net=net.double()
    model = dde.Model(data, net)
    model.compile(optimizer, lr=lr, metrics=metrics)
    if not resample:
        losshistory, train_state = model.train(iterations=iterations)
    else:
        pde_resampler = dde.callbacks.PDEPointResampler(period=period)
        losshistory, train_state = model.train(iterations=iterations, callbacks=[pde_resampler])
    save_model(model)
    return model, losshistory, train_state


def train_model_LBFGS(model, metrics=None,
                      resample=False, period=None):
    """
    进一步训练模型，使用L-BFGS优化器，一般在train_model训练之后再使用，以提高训练精度。
    参数:
        model: 训练好的模型
        metrics: 评估指标
        resample: 是否重采样
        period: 重采样周期
    返回:
        model: 训练好的模型
        losshistory: 损失历史
        train_state: 训练状态
    """
    # model.compile("L-BFGS-B")
    model.compile("L-BFGS", metrics=metrics)
    if not resample:
        losshistory, train_state = model.train()
    else:
        pde_resampler = dde.callbacks.PDEPointResampler(period=period)
        losshistory, train_state = model.train(callbacks=[pde_resampler])
    save_model(model)
    return model, losshistory, train_state

def save_model(model):
    save_path = os.path.join(os.environ["OUTPUT_DIR"], "model.pt")
    checkpoint = {
        "model_state_dict": model.net.state_dict(),
        "optimizer_state_dict": model.opt.state_dict(),
    }
    torch.save(checkpoint, save_path)
    
def load_model(net_path=None, model_path=None):
    if net_path is None:
        net_path = os.path.join(os.environ["OUTPUT_DIR"], "net_config.json")
    if model_path is None:
        model_path = os.path.join(os.environ["OUTPUT_DIR"], "model.pt")
    with open(net_path, "r", encoding="utf-8") as f:
        net_config = json.load(f)
    load_net = dde.nn.FNN(
        **net_config
    )
    load_model = dde.Model(None, load_net)
    load_model.compile(optimizer="adam", lr=0.00)   # 参数任意
    # 加载
    checkpoint = torch.load(model_path, weights_only=True)
    load_model.net.load_state_dict(checkpoint["model_state_dict"])
    return load_model
    

# 工具9：可视化和保存结果
# 工具9调用的子函数1，绘制损失曲线
def plot_loss2(loss_history):
    fig, ax = plt.subplots(figsize=(10, 5))
    # plt.figure(figsize=(10, 5))  # 明确创建新 Figure
    loss_train = np.sum(loss_history.loss_train, axis=1)
    loss_test = np.sum(loss_history.loss_test, axis=1)

    ax.semilogy(loss_history.steps, loss_train, label="Train loss")
    ax.semilogy(loss_history.steps, loss_test, label="Test loss")

    if hasattr(loss_history, 'metrics_test') and loss_history.metrics_test:
        for i in range(len(loss_history.metrics_test[0])):
            plt.semilogy(
                loss_history.steps,
                np.array(loss_history.metrics_test)[:, i],
                label=f"Test metric {i + 1}",
            )

    ax.xlabel("# Steps")
    ax.legend()
    ax.grid(True)
    # fig1 = plt.gcf()  # ✅ 正确获取当前 Figure
    # plt.show()
    return (fig, ax)

def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))  # 明确创建新 Figure
    loss_train = np.sum(loss_history.loss_train, axis=1)
    loss_test = np.sum(loss_history.loss_test, axis=1)

    plt.semilogy(loss_history.steps, loss_train, label="Train loss")
    plt.semilogy(loss_history.steps, loss_test, label="Test loss")

    if hasattr(loss_history, 'metrics_test') and loss_history.metrics_test:
        for i in range(len(loss_history.metrics_test[0])):
            plt.semilogy(
                loss_history.steps,
                np.array(loss_history.metrics_test)[:, i],
                label=f"Test metric {i + 1}",
            )

    plt.xlabel("# Steps")
    plt.legend()
    plt.grid(True)
    fig1 = plt.gcf()  # ✅ 正确获取当前 Figure
    # plt.show()
    return fig1


# 工具9调用的子函数2，绘制最好状态
def plot_Best_state2(train_state):
    y_train, y_test, best_y, best_ystd = _pack_data(train_state)
    y_dim = best_y.shape[1]
    # fig2 = None
    fig, ax = None, None
    if train_state.X_test.shape[1] == 1:
        idx = np.argsort(train_state.X_test[:, 0])
        X = train_state.X_test[idx, 0]

        # plt.figure(figsize=(10, 5))  # 显式创建新 Figure
        fig, ax = plt.subplots(figsize=(10, 5))  # 显式创建新 Figure
        for i in range(y_dim):
            if y_train is not None:
                ax.plot(train_state.X_train[:, 0], y_train[:, i], "ok", label="Train")
            if y_test is not None:
                ax.plot(X, y_test[idx, i], "-k", label="True")
            plt.plot(X, best_y[idx, i], "--r", label="Prediction")
            if best_ystd is not None:
                ax.plot(X, best_y[idx, i] + 1.96 * best_ystd[idx, i], "-b", label="95% CI")
                ax.plot(X, best_y[idx, i] - 1.96 * best_ystd[idx, i], "-b")
        ax.xlabel("x")
        ax.ylabel("y")
        ax.legend()
        # fig2 = plt.gcf()  # ✅ 获取当前 Figure
        # plt.show()

    # 2D 绘图
    elif train_state.X_test.shape[1] == 2:
        # plt.figure(figsize=(10, 5))
        # ax = plt.axes(projection="3d")  # ✅ 正确 3D 语法
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': '3d'})  # 显式创建新 Figure
        for i in range(y_dim):
            ax.plot3D(
                train_state.X_test[:, 0],
                train_state.X_test[:, 1],
                best_y[:, i],
                ".",
                label=f"$y_{i + 1}$"
            )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$y$")
        ax.legend()
        # fig2 = plt.gcf()
        # plt.show()

    return (fig, ax)

def plot_Best_state(train_state):
    y_train, y_test, best_y, best_ystd = _pack_data(train_state)
    y_dim = best_y.shape[1]
    fig2 = None
    if train_state.X_test.shape[1] == 1:
        idx = np.argsort(train_state.X_test[:, 0])
        X = train_state.X_test[idx, 0]

        plt.figure(figsize=(10, 5))  # 显式创建新 Figure
        for i in range(y_dim):
            if y_train is not None:
                plt.plot(train_state.X_train[:, 0], y_train[:, i], "ok", label="Train")
            if y_test is not None:
                plt.plot(X, y_test[idx, i], "-k", label="True")
            plt.plot(X, best_y[idx, i], "--r", label="Prediction")
            if best_ystd is not None:
                plt.plot(X, best_y[idx, i] + 1.96 * best_ystd[idx, i], "-b", label="95% CI")
                plt.plot(X, best_y[idx, i] - 1.96 * best_ystd[idx, i], "-b")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        fig2 = plt.gcf()  # ✅ 获取当前 Figure
        # plt.show()

    # 2D 绘图
    elif train_state.X_test.shape[1] == 2:
        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection="3d")  # ✅ 正确 3D 语法
        for i in range(y_dim):
            ax.plot3D(
                train_state.X_test[:, 0],
                train_state.X_test[:, 1],
                best_y[:, i],
                ".",
                label=f"$y_{i + 1}$"
            )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$y$")
        ax.legend()
        fig2 = plt.gcf()
        # plt.show()

    return fig2

def save_loss_history(loss_history, fname):
    """Save the training and testing loss history to a .dat file."""
    print("Saving loss history to {} ...".format(fname))
    
    # 将每个步骤的多维数据转换为字符串
    loss_train_str = [','.join(map(str, arr)) for arr in loss_history.loss_train]
    loss_test_str = [','.join(map(str, arr)) for arr in loss_history.loss_test]
    metrics_test_str = [','.join(map(str, arr)) for arr in loss_history.metrics_test]
    
    # 将所有数据组合成一个二维数组，并保存到文件
    data = np.column_stack((
        np.array(loss_history.steps),
        np.array(loss_train_str, dtype=np.str_),
        np.array(loss_test_str, dtype=np.str_),
        np.array(metrics_test_str, dtype=np.str_)
    ))
    
    np.savetxt(fname, data, fmt='%s', delimiter='\t', header="steps\tloss_train\tloss_test\tmetrics_test")
    # np.savetxt(fname, loss, header="step, loss_train, loss_test, metrics_test")
    
def load_loss_history(fname):
    """Load the training and testing loss history from a .dat file."""
    print("Loading loss history from {} ...".format(fname))
    
    # 读取数据
    data = np.genfromtxt(fname, delimiter='\t', skip_header=1, dtype=None, encoding='utf-8')
    
    # 提取各列
    steps = data['f0'].astype(int)
    loss_train = [np.fromstring(s, sep=',') for s in data['f1']]
    loss_test = [np.fromstring(s, sep=',') for s in data['f2']]
    metrics_test = [np.fromstring(s, sep=',') for s in data['f3']]
    
    loaded_history = {
        'steps': steps,
        'loss_train': loss_train,
        'loss_test': loss_test,
        'metrics_test': metrics_test
    }
    
    return loaded_history

# 工具9：可视化和保存结果
def visualize_and_save(
        loss_history,
        train_state,
        is_save: bool = True,
        is_plot: bool = True,
):
    """
        可视化训练结果并可选保存。
        参数:
            loss_history: 损失历史
            train_state: 训练状态
            is_save: 是否保存
            is_plot: 是否绘图
        """
    output_dir = os.getenv("OUTPUT_DIR") if os.getenv("OUTPUT_DIR") else os.getcwd()
    fig1, fig2 = None, None

    # 确保目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 保存数据
    if is_save:
        loss_fname = os.path.join(output_dir, "loss.dat")
        train_fname = os.path.join(output_dir, "train.dat")
        test_fname = os.path.join(output_dir, "test.dat")
        save_loss_history(loss_history, loss_fname)
        dde.utils.save_best_state(train_state, train_fname, test_fname)

    # 绘图
    if is_plot:
        # 第一幅图：损失曲线
        fig1 = plot_loss(loss_history)
        # 第二幅图：预测结果
        fig2 = plot_Best_state(train_state)
        
        # TODO: save figures to OUTPUT_DIR
        plt.figure(fig1.figure)
        plt.savefig(os.path.join(os.getenv('OUTPUT_DIR'), f"visualize_and_save_1.png"))
        plt.figure(fig2.figure)
        plt.savefig(os.path.join(os.getenv('OUTPUT_DIR'), f"visualize_and_save_2.png"))

    return fig1, fig2
    


if __name__ == "__main__":
    equation = "\\Delta u = 2, The spatial dimension is one-dimensional"
    # equation = "-\\nabla^2 u + u + 0.6v = -2\\pi^2\\sin(\\pi x)\\sin(\\pi y), -\\nabla^2 v - 0.5u + 0.8v = -8\\pi^2\\sin(2\\pi x)\\sin(2\\pi y)"
    pde_func, pde_code = define_pde(equation)
    print(pde_func)
    print("The PDE function is defined as follows:\n" + pde_code)
    #
    # ref_sol = "u(x) = (x+1)^2"
    # ref_func, ref_code = define_reference_solution(ref_sol)
    # print("The reference solution is defined as follows:\n"+ref_code)
    #

    # geomtime = define_domain("Polygon", [[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]], is_time_dependent=True, time_range=[0, 1])
    # print(geomtime)
    # bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    # ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)
    # net = create_network(input_dim=2, output_dim=1, hidden_layers=[20, 20, 20], activation="tanh")
    # data = create_training_data(geomtime, pde_func, 1000, 100, num_initial=100, num_test=1000, bc=[bc],ic=[ic],is_time_dependent=True)
    # print(data)
    # bc_description = "This is a time-independent PDE with one-dimensional space. The range of x is [-1,1]. The left boundary satisfies the Dirichlet boundary condition: u(-1)=0; the right boundary satisfies the Neumann boundary condition\\left.\\dfrac{du}{dx}\\right|_{x=1} =4"
    # bc,bc_code = define_boundary_condition(geom, bc_description)
    # print("The boundary conditions are defined as follows:\n"+bc_code)
    #
    # data = create_training_data(geom, pde_func, 1000, 100, num_initial=100, num_test=1000, bc=bc, ref_func=ref_func)
    #
    # net = create_network(input_dim=1, output_dim=1, hidden_layers=[20, 20, 20], activation="tanh")
    #
    # model, losshistory, train_state = train_model(data, net, optimizer="adam", lr=0.001, iterations=10000, metrics=["l2 relative error"])
    #
    # visualize_and_save(losshistory, train_state, is_save=False, is_plot=True)
    #
    # print("The training is done.")

    # ic_description = "This is a time-independent PDE with one-dimensional space. The range of x is [-1,1] and the range of time is [0, 1].  The initial condition is u(x,0)=\\sin(\\pi * x) and \\frac{\\partial u}{\\partial t}\\bigg|_{t=0} = \\cos(\\pi * x) "
    # ic, ic_code = define_initial_condition(geomtime, ic_description)
    # print("The initial conditions are defined as follows:\n" + ic_code)
    # print(ic)
    
    # fname = "./loss.dat"
    # loaded_loss_history = load_loss_history(fname)
    # # 打印加载的数据
    # print("Steps:", loaded_loss_history['steps'])
    # print("Train Loss:", loaded_loss_history['loss_train'])
    # if loaded_loss_history['loss_test'] is not None:
    #     print("Test Loss:", loaded_loss_history['loss_test'])
    # if loaded_loss_history['metrics_test'] is not None:
    #     print("Test Metrics:", loaded_loss_history['metrics_test'])
    # print(len(loaded_loss_history['steps'])==len(loaded_loss_history['loss_train']))
