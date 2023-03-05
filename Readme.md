# **QOSF Cohort 7**

## Task 3 - QSVM

### Submission by Harishankar P V

### Dated: 05 March 2023

**Problem Statement:** Generate a Quantum Support Vector Machine (QSVM) using the iris dataset and try to
propose a kernel from a parametric quantum circuit to classify the three classes(setosa,
versicolor, virginica) using the one-vs-all format, the kernel only works as binary
classification. Identify the proposal with the lowest number of qubits and depth to obtain
higher accuracy. You can use the UU† format or using the Swap-Test.

**NOTE: Refer the _qml_qsvm_notes.pdf_ in the same github repo for my handwritten notes. The content of the pdf is part of my Quantum Machine Learning notes, originally taken from Qiskit Summer School 2021 lectures. It's purpose is to given an idea to the mentors of my current understanding of QML and QSVM in particular. I have reproduced my handwritten notes as pdf specifically for this purpose.**

# **Quantum Kernels and Support Vector Machines**

## 1. Data Encoding

We will take the classical data and encode it to the quantum state space using a quantum feature map. The choice of which feature map to use is important and may depend on the given dataset we want to classify. Here we'll look at the feature maps available in Qiskit, before selecting and customising one to encode our data.

### 1.1. Quantum Feature Maps

As the name suggests, a quantum feature map $\phi(\mathbf{x})$ is a map from the classical feature vector $\mathbf{x}$ to the quantum state $|\Phi(\mathbf{x})\rangle\langle\Phi(\mathbf{x})|$. This is faciliated by applying the unitary operation $\mathcal{U}_{\Phi(\mathbf{x})}$ on the initial state $|0\rangle^{n}$ where _n_ is the number of qubits being used for encoding.

The feature maps currently available in Qiskit ([`PauliFeatureMap`](https://qiskit.org/documentation/stubs/qiskit.circuit.library.PauliFeatureMap.html), [`ZZFeatureMap`](https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZFeatureMap.html) and [`ZFeatureMap`](https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZZFeatureMap.html)) are those introduced in [_Havlicek et al_. Nature **567**, 209-212 (2019)](https://www.nature.com/articles/s41586-019-0980-2), in particular the `ZZFeatureMap` is conjectured to be hard to simulate classically and can be implemented as short-depth circuits on near-term quantum devices.

The `PauliFeatureMap` is defined as:

```python
PauliFeatureMap(feature_dimension=None, reps=2,
                entanglement='full', paulis=None,
                data_map_func=None, parameter_prefix='x',
                insert_barriers=False)
```

and describes the unitary operator of depth $d$:

$$ \mathcal{U}_{\Phi(\mathbf{x})}=\prod_d U_{\Phi(\mathbf{x})}H^{\otimes n},\ U*{\Phi(\mathbf{x})}=\exp\left(i\sum*{S\subseteq[n]}\phi*S(\mathbf{x})\prod*{k\in S} P_i\right), $$

which contains layers of Hadamard gates interleaved with entangling blocks, $U_{\Phi(\mathbf{x})}$, encoding the classical data as shown in circuit diagram below for $d=2$.

![Alt text](https://learn.qiskit.org/content/quantum-machine-learning/images/kernel/featuremap.svg "a title")

Within the entangling blocks, $U_{\Phi(\mathbf{x})}$: $P_i \in \{ I, X, Y, Z \}$ denotes the Pauli matrices, the index $S$ describes connectivities between different qubits or datapoints: $S \in \{\binom{n}{k}\ combinations,\ k = 1,... n \}$, and by default the data mapping function $\phi_S(\mathbf{x})$ is

$$
\phi_S:\mathbf{x}\mapsto \Bigg\{\begin{array}{ll}
    x_i & {if}\ S=\{i\} \\
        (\pi-x_i)(\pi-x_j) & {if}\ S=\{i,j\}
    \end{array}
$$

when $k = 1, P_0 = Z$, this is the `ZFeatureMap`:
$$
\mathcal{U}_{\Phi(\mathbf{x})} = \left( \exp\left(i\sum_j \phi_{\{j\}}(\mathbf{x}) \, Z_j\right) \, H^{\otimes n} \right)^d.
$$

which is defined as:

```python
ZFeatureMap(feature_dimension, reps=2,
            data_map_func=None, insert_barriers=False)
```

and when $k = 2, P_0 = Z, P_1 = ZZ$, this is the `ZZFeatureMap`:
$$\mathcal{U}_{\Phi(\mathbf{x})} = \left( \exp\left(i\sum_{jk} \phi_{\{j,k\}}(\mathbf{x}) \, Z_j \otimes Z_k\right) \, \exp\left(i\sum_j \phi_{\{j\}}(\mathbf{x}) \, Z_j\right) \, H^{\otimes n} \right)^d.$$

which is defined as:

```python
ZZFeatureMap(feature_dimension, reps=2,
             entanglement='full', data_map_func=None,
             insert_barriers=False)
```

We can also customise the Pauli gates in the feature map, for example, $P_0 = X, P_1 = Y, P_2 = ZZ$:

$$
\mathcal{U}_{\Phi(\mathbf{x})} = \left( \exp\left(i\sum_{jk} \phi_{\{j,k\}}(\mathbf{x}) \, Z_j \otimes Z_k\right) \, \exp\left(i\sum_{j} \phi_{\{j\}}(\mathbf{x}) \, Y_j\right) \, \exp\left(i\sum_j \phi_{\{j\}}(\mathbf{x}) \, X_j\right) \, H^{\otimes n} \right)^d.
$$

We can also define a custom data mapping function, for example:

$$
\phi_S:\mathbf{x}\mapsto \Bigg\{\begin{array}{ll}
    x_i &  {if}\ S=\{i\} \\
        \sin(\pi-x_i)\sin(\pi-x_j) & {if}\ S=\{i,j\}
    \end{array}
$$

The [`NLocal`](https://qiskit.org/documentation/stubs/qiskit.circuit.library.NLocal.html) and [`TwoLocal`](https://qiskit.org/documentation/stubs/qiskit.circuit.library.TwoLocal.html) functions in Qiskit's circuit library can also be used to create parameterized quantum circuits as feature maps.

```python
TwoLocal(num_qubits=None, reps=3, rotation_blocks=None,
         entanglement_blocks=None, entanglement='full',
         skip_unentangled_qubits=False,
         skip_final_rotation_layer=False,
         parameter_prefix='θ', insert_barriers=False,
         initial_state=None)
```

```python
NLocal(num_qubits=None, reps=1, rotation_blocks=None,
       entanglement_blocks=None, entanglement=None,
       skip_unentangled_qubits=False,
       skip_final_rotation_layer=False,
       overwrite_block_parameters=True,
       parameter_prefix='θ', insert_barriers=False,
       initial_state=None, name='nlocal')
```

Both functions create parameterized circuits of alternating rotation and entanglement layers. In both layers, parameterized circuit-blocks act on the circuit in a defined way. In the rotation layer, the blocks are applied stacked on top of each other, while in the entanglement layer according to the entanglement strategy. Each layer is repeated a number of times, and by default a final rotation layer is appended.

In `NLocal`, the circuit blocks can have arbitrary sizes (smaller equal to the number of qubits in the circuit), while in `TwoLocal`, the rotation layers are single qubit gates applied on all qubits and the entanglement layer uses two-qubit gates.

## 2. Quantum Kernel Estimation

A quantum feature map, $\phi(\mathbf{x})$, naturally gives rise to a quantum kernel, $k(\mathbf{x}_i,\mathbf{x}_j)= \phi(\mathbf{x}_j)^\dagger\phi(\mathbf{x}_i)$, which can be seen as a measure of similarity: $k(\mathbf{x}_i,\mathbf{x}_j)$ is large when $\mathbf{x}_i$ and $\mathbf{x}_j$ are close.

When considering finite data, we can represent the quantum kernel as a matrix:
$K_{ij} = \left| \langle \phi^\dagger(\mathbf{x}_j)| \phi(\mathbf{x}_i) \rangle \right|^{2}$. We can calculate each element of this kernel matrix on a quantum computer by calculating the transition amplitude:

$$
\left| \langle \phi^\dagger(\mathbf{x}_j)| \phi(\mathbf{x}_i) \rangle \right|^{2} =
\left| \langle 0^{\otimes n} | \mathbf{U_\phi^\dagger}(\mathbf{x}_j) \mathbf{U_\phi}(\mathbf{x_i}) | 0^{\otimes n} \rangle \right|^{2}
$$

assuming the feature map is a parameterized quantum circuit, which can be described as a unitary transformation $\mathbf{U_\phi}(\mathbf{x})$ on $n$ qubits.

This provides us with an estimate of the quantum kernel matrix, which we can then use in a kernel machine learning algorithm, such as support vector classification.

As discussed in [_Havlicek et al_. Nature 567, 209-212 (2019)](https://www.nature.com/articles/s41586-019-0980-2), quantum kernel machine algorithms only have the potential of quantum advantage over classical approaches if the corresponding quantum kernel is hard to estimate classically.

As we will see later, the hardness of estimating the kernel with classical resources is of course only a necessary and not always sufficient condition to obtain a quantum advantage.

However, it was proven recently in [_Liu et al._ arXiv:2010.02174 (2020)](https://arxiv.org/abs/2010.02174) that learning problems exist for which learners with access to quantum kernel methods have a quantum advantage over allclassical learners.

## 3. Quantum Support Vector Classification

Introduced in [_Havlicek et al_. Nature 567, 209-212 (2019)](https://www.nature.com/articles/s41586-019-0980-2), the quantum kernel support vector classification algorithm consists of these steps:

![Alt text](https://learn.qiskit.org/content/quantum-machine-learning/images/kernel/qsvc.svg "a title")

1. Build the train and test quantum kernel matrices.

2. For each pair of datapoints in the training dataset $\mathbf{x}_{i},\mathbf{x}_j$, apply the feature map and measure the transition probability:

   $$
    K*{ij} = \left| \langle 0 | \mathbf{U}^\dagger*{\Phi(\mathbf{x*j})} \mathbf{U}*{\Phi(\mathbf{x_i})} | 0 \rangle \right|^2.
   $$

3. For each training datapoint $\mathbf{x_i}$ and testing point $\mathbf{y_i}$, apply the feature map and measure the transition probability:

   $$
   K*{ij} = \left| \langle 0 | \mathbf{U}^\dagger*{\Phi(\mathbf{y*i})} \mathbf{U}*{\Phi(\mathbf{x_i})} | 0 \rangle \right|^2.
   $$

4. Use the train and test quantum kernel matrices in a classical support vector machine classification algorithm.

The `scikit-learn` `svc` algorithm allows us to define a [custom kernel](https://scikit-learn.org/stable/modules/svm.html#custom-kernels) in two ways: by providing the kernel as a callable function or by precomputing the kernel matrix. We can do either of these using the `QuantumKernel` class in Qiskit.
