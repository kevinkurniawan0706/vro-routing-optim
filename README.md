# 🚀 fleet_v3 — HCVRPTW with Three Heterogeneous Vehicles

PyTorch implementation of attention-based reinforcement learning policies for the **Heterogeneous Capacitated Vehicle Routing Problem with Time Windows (HCVRPTW)**. The model supports three vehicle types, customer time windows, and both **min-max** or **min-sum** optimization objectives. Training is performed using the **REINFORCE** algorithm with a greedy **rollout** baseline.

This project is structured for flexible execution, supporting command-line training/evaluation as well as an interactive **Google Colab** pipeline that replaces the legacy `Makefile` structure.

---

## 📋 Table of Contents

1. [About the Model](#-about-the-model)
2. [Folder & Repository Layout](#-folder--repository-layout)
3. [Requirements](#-requirements)
4. [Google Colab Interactive Usage](#-google-colab-interactive-usage)
5. [Manual Command Line Interface (CLI) Usage](#-manual-command-line-interface-cli-usage)
6. [Outputs, Checkpoints, & Logging](#-outputs-checkpoints--logging)
7. [Citation](#-citation)

---

## 🧠 About the Model

The problem integrates three core optimization challenges:
* **Heterogeneous Fleet:** Supports three vehicle types with different capacities.
* **Capacitated Routing:** Demands must respect vehicle capacities.
* **Time Windows:** Customers can only be served within specific intervals; time windows are represented using a 24-dimensional vector encoding representing hourly shifts.

### Neural Architecture
The policy uses an **encoder–decoder** attention architecture (HCVRPTW-Transformer):
1. **Graph Encoder:** Builds node embeddings of the customers and depot.
2. **Decoder:** Autoregressively chooses **which vehicle** to deploy next and **which customer** it should visit.
3. **Multi-Decoder Setup:** Trains several decoder heads simultaneously to encourage **diverse route candidates** with a **Kullback–Leibler (KL) divergence** regularization term between first-step path distributions. During inference, the best decoded path can be selected based on cost.

### Implemented Model Variants

All policy variants share the unified module [attention_model.py](file:///D:/MGD/Project/Research/vro-routing-optim/core/nets/attention_model.py):

| `--model` | Description |
| :--- | :--- |
| `attention` | **Default:** Time window encoded in node embeddings; single decoder head. |
| `attention_explicit` | Vehicle-context speeds for explicit time window handling in vehicle selection. |
| `attention_implicit` | Shorter node features (time window is not concatenated in initialization embedding). |
| `attention_decoder` | Time-window features fused directly in decoder pre-computation. |
| `attention_multi_decoder` | Several decoders/paths; must be trained using `training/run_multi.py`. |

---

## 📂 Folder & Repository Layout

The project has been reorganized into a clean, modular structure. Below is the tree layout:

```text
vro-routing-optim/
├── colab_data_custom.ipynb       # Colab notebook for custom data generation
├── colab_data_synthetic.ipynb    # Colab notebook for synthetic data generation
├── colab_eval.ipynb              # Colab notebook for step-by-step evaluation
├── colab_eval_test.ipynb         # Colab notebook for testing evaluation
├── colab_master_pipeline.ipynb   # Master Colab pipeline (replaces legacy Makefile)
├── colab_run.ipynb               # Step-by-step interactive training analysis
├── colab_run_multi.ipynb         # Step-by-step interactive multi-decoder training
├── README.md                     # Project documentation (this file)
│
├── core/                         # Core neural network models and problems
│   ├── eval/
│   │   └── eval.py               # Dataset evaluation and inference script
│   ├── nets/
│   │   ├── attention_model.py    # Unified AttentionHCVRPActor models
│   │   ├── critic_network.py     # Critic network for baseline estimation
│   │   ├── graph_encoder.py      # Attention-based encoder network
│   │   └── pointer_network.py    # Pointer network architecture
│   ├── problems/
│   │   └── hcvrp/
│   │       ├── problem_hcvrp.py  # HCVRP problem definition and costs
│   │       └── state_hcvrp.py    # HCVRP environment state transitions
│   ├── train.py                  # Inner training epoch loops and validation steps
│   └── reinforce_baselines.py    # Inner Reinforce baseline classes
│
├── training/                     # Main training and execution runner
│   ├── run.py                    # Main CLI entry point for single-decoder models
│   ├── run_multi.py              # Main CLI entry point for multi-decoder models
│   ├── options.py                # Command-line options and parameters parser
│   ├── train.py                  # Training loops setup
│   └── reinforce_baselines.py    # Baseline classes configuration
│
├── data_generator/               # Dataset generation helpers
│   ├── custom/
│   │   ├── custom_data_generator.py # Custom coordinate & demand generator
│   │   └── custom_data_payload.json # Custom data templates
│   └── synthetic/
│       ├── generate_data.py      # Main synthetic data generator
│       ├── generate_5_set_data.py# 5-vehicle synthetic data generator
│       └── single_data_generator.py # Single instance generator
│
├── utils/                        # Common helper utilities
│   ├── functions.py              # load_model, load_problem, sample_many helpers
│   ├── data_utils.py             # Data saving/loading helpers
│   ├── beam_search.py            # Beam search implementation
│   ├── boolmask.py               # Boolean masking operations
│   └── log_utils.py              # TensorBoard logging helpers
│
├── test/                         # Visualizations and test code
│   └── visualize.py              # Route visualization scripts
│
├── data/                         # Saved data pickles
│   └── hcvrp/                    # Pre-generated HCVRP benchmark datasets
│
└── outputs/                      # Saved models, parameters, checkpoints & logs
```

---

## 🛠️ Requirements

Install the dependencies from the repository root:
```bash
pip install -r requirements.txt
```
Key required libraries:
* PyTorch (CUDA-capable recommended for training)
* `tensorboard_logger`
* `tqdm`
* `numpy`
* `scipy`
* `matplotlib`

---

## 📓 Google Colab Interactive Usage

For Google Colab or local Jupyter environments, use the Jupyter Notebooks located in the root directory. They contain step-by-step guides, interactive code cells, and manual loops to run the project.

* **`colab_master_pipeline.ipynb`**: The master notebook that replaces the legacy `Makefile`. Run environment setup, data generation, smoke tests, full model training, and evaluation directly from this notebook.
* **`colab_run.ipynb` / `colab_run_multi.ipynb`**: Detailed step-by-step training and manual loops for single-decoder and multi-decoder architectures, explaining model and baseline initializations.
* **`colab_eval.ipynb` / `colab_eval_test.ipynb`**: Step-by-step model evaluation, validation, and route visualization analysis.
* **`colab_data_custom.ipynb` / `colab_data_synthetic.ipynb`**: Step-by-step data generation for custom demands and synthetic instances.

---

## 💻 Manual Command Line Interface (CLI) Usage

All scripts should be executed from the project root directory.

### 1. Generate Dataset

* **Synthetic Data Generation** (e.g. 40 customer graph, 3 vehicles):
  ```bash
  python data_generator/synthetic/generate_data.py --veh_num 3 --graph_size 40
  ```

* **Custom Data Generation** (based on custom coordinate and demand templates):
  ```bash
  python data_generator/custom/custom_data_generator.py
  ```

### 2. Train Model

* **Single-Decoder Models** (e.g. `attention_decoder` model on 40-node graph with rollout baseline):
  ```bash
  python training/run.py --graph_size 40 --baseline rollout --run_name 'my_run' --obj min-max --model attention_decoder
  ```

* **Multi-Decoder Model** (e.g. 5 paths, `attention_multi_decoder` model with KL divergence loss):
  ```bash
  python training/run_multi.py --graph_size 40 --baseline rollout --run_name 'my_multi' --obj min-max --model attention_multi_decoder --n_paths 5 --kl-loss 1.0
  ```

> [!TIP]
> Use `--no_cuda` if you wish to run/test on CPU. To limit training to specific GPUs, use `CUDA_VISIBLE_DEVICES=0`.

### 3. Resume / Warm-Start Training

To resume from a saved checkpoint, use the `--resume` option:
```bash
python training/run.py --graph_size 40 --model attention_decoder --baseline rollout --resume outputs/hcvrp_40/my_run/epoch-49.pt
```

### 4. Evaluate Checkpoint

Evaluate a model checkpoint on a saved dataset using a decoding strategy (`greedy` or `sample`):

* **Greedy Decoding:**
  ```bash
  python core/eval/eval.py data/hcvrp/hcvrp_v3_40_seed99999_tw.pkl --model outputs/hcvrp_40/my_run/epoch-49.pt --decode_strategy greedy --obj min-max
  ```

* **Sample Decoding** (e.g., sample 1280 routes per instance):
  ```bash
  python core/eval/eval.py data/hcvrp/hcvrp_v3_40_seed99999_tw.pkl --model outputs/hcvrp_40/my_run/epoch-49.pt --decode_strategy sample --width 1280 --eval_batch_size 1 --obj min-max
  ```

---

## 💾 Outputs, Checkpoints, & Logging

* **Checkpoints:** Training weights are saved under `outputs/hcvrp_<graph_size>/<run_name>/` as `epoch-<num>.pt`.
* **Configurations:** The exact arguments are saved in `args.json` in the run directory. This file is automatically read during evaluation or resuming.
* **Logs:** Tensorboard logs are saved under the directory specified by `--log_dir` (default: `logs/hcvrp_<graph_size>/<run_name>`). View logs by running `tensorboard --logdir logs/`.

---

## 📄 Citation

If you use this code in your research, please cite the original HCVRP baseline paper:

```bibtex
@article{li2021hcvrp,
  title={Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem},
  author={Li, Jingwen and Ma, Yining and Gao, Ruize and Cao, Zhiguang and Lim, Andrew and Song, Wen and Zhang, Jie},
  journal={IEEE Transactions on Cybernetics},
  volume={52},
  number={12},
  pages={13572--13585},
  year={2022},
  publisher={IEEE},
  doi={10.1109/TCYB.2021.3111082}
}
```
