<div align="center">
<h1 align="center"> PDE-Agent: A toolchain-augmented multi-agent framework for PDE solving </h1>

</div>

## 📌 Contents

- [⌚️ Overview](#2)
- [📦 Project Framework](#3)
- [⚡ Getting Started](#4)
  - [🔧️ Installation](#4.1)
  - [🚀 Quick Start](#4.2)
- [📊 Data Preparation](#5)
- [📝️ Cite](#6)

## 🆕 News

- **[2025-12-18]**: 🎉 PDE-Agent is Now Available in This Repository!
- **[2025-11]**: The PDE-Agent project will available on GitHub.


<h2 id="2">⌚️  Overview</h2>

In this work, we frame PDE solving as tool invocation via LLM-driven agents and introduce PDE-Agent, the first toolchain-augmented multiagent collaboration framework, inheriting the reasoning capacity of LLMs and the controllability of external tools and enabling automated PDE solving from natural language descriptions.

PDE-Agent leverages the strengths of multi-agent and multi-tool collaboration through two key innovations: A Prog-Act framework with graph memory and A Resource-Pool integrated with a tool-parameter separation mechanism.



### Key Features:

- **PDE Tools**: The first LLM-driven multiagent framework equipped with modular PDE-toolkits rigorously encapsulated for automated PDEs solving.
- **Prog-Act**: A Prog-Act framework with graph memory for multi-agent
  collaboration, which enables effective dynamic planning and
  error correction via dual-loop mechanisms (localized fixes
  and global revisions).
- **Resource-Pool**: A Resource-Pool integrated with a tool-parameter separation mechanism for multi-tool collaboration. This centralizes the management of runtime artifacts and resolves inter-tool dependency gaps in existing frameworks.



<h2 id="3">📦 Project Framework</h2>

```plaintext
/SciToolAgent
├── data              # Data storage directory
├── scripts           # Scripts for running PDE-Agent
├── test              # Testing scripts
├── SciToolEval       # SciToolEval related files
└── HiveMinds         # PDE-Agent main files
    ├── minds     	  # Multi-Agent
    ├── toolkits      # PDE tools
    ├── memory        # Memory and resources pool
    ├── utils         # Common utility functions and toolchain
    ├── hive_mind.py  # The core logic of PDE-Agent
    └── .env		  # YOUR API KEY
```

<h2 id="4">⚡ Getting Started</h2>

<h3 id="4.1">🔧️ Installation</h2>

1. **Clone the repository**  
   First, clone the project to your local machine:


   ```bash
   git clone https://github.com/xxxxxx.git
   cd PDE-Agent
   ```

2. **Create and activate a virtual environment**  
   Set up a new virtual environment using Conda and activate it:
   
    ```bash
    conda create -n PDEA python=3.9
    conda activate PDEA
    ```
3. **Install project dependencies**
   Install the necessary dependencies for the project:
    ```bash
    pip install -r ./pyenv/requirements.txt
    ```
   


<h3 id="4.2">🚀Quick Start</h2>
You need to modify the `.env` files to set your `API_KEY` and `API_BASE`.
* 
    ```
    DEEPSEEK_API_KEY=YOUR_DEEPSEEK_API_KEY
    DEEPSEEK_BASE_URL=DEEPSEEK_BASE_URL
    ```

1.  Load dependence.
    ```bash
    cd pyenv
    bash deps.sh
    ```

2.  Run the PDE-Agent
    ```bash
    cd ../scripts
    bash run.sh
    ```
    You can also give the `case_file` or edit `run.sh`:
    
    ```bash
    bash run.sh ./data/yaml/inputs/Allen-Cahn equation/eval1.yaml
    ```
    
    The results will be saved in `./scripts/logs`
    
    

<h2 id="5">📊 Data Preparation</h2>

🔄 Data is in Progress!
We are currently performing **data cleaning, standardization, and validation** for the dataset of this project:

- Eliminating redundant, incomplete, or invalid data entries
- Unifying data formats to ensure consistency across the dataset
- Verifying data accuracy to support reliable subsequent use

The data preparation work is ongoing, and we will update the processed dataset, along with relevant supporting materials, in this repository once this process is completed.

<h2 id="8">📝️  Cite</h2>

```
@article{}
```
