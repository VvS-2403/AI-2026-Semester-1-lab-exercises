# Neural Models Lab Submission Report

**Student Name:** Vismay
**Course:** Artificial Intelligence

## 1. Problem Specification and Linear-Separability Explanation

### Task 1: Understand the problem before coding
*   **Input Space $\mathcal{X}$**: `{(0, 0), (0, 1), (1, 0), (1, 1)}`
*   **Output Space $\mathcal{Y}$**: `{0, 1}` (Disagreement Warning)
*   **Labelled Examples**:
    *   (0, 0) $\rightarrow$ 0
    *   (0, 1) $\rightarrow$ 1
    *   (1, 0) $\rightarrow$ 1
    *   (1, 1) $\rightarrow$ 0

*(Sketch of the four points in the $x_1, x_2$ plane should go here)*

**Why one straight decision boundary cannot separate the two classes:**
The points for class 1 (0,1 and 1,0) and class 0 (0,0 and 1,1) are positioned at diagonally opposite corners of a square. No single straight line can separate the two corners with class 1 from the two corners with class 0 without intersecting the other class.

**Prediction for a single affine transformation followed by a sigmoid output:**
Since a single affine transformation followed by a sigmoid output is a linear classifier, it will not be able to solve the XOR problem. The model will likely converge to predicting 0.5 for all inputs (or similar compromise values), resulting in high loss and incorrect predictions.

---

## 2. Model Design and Validation Criteria

### Task 2: Design the intelligent agent
*   **Architecture**: 2 inputs $\rightarrow$ 2 hidden units $\rightarrow$ 1 output
*   **Hidden Activation**: Sigmoid (initially)
*   **Output**: Sigmoid + Binary Cross-Entropy
*   **Optimization**: Gradient-based

**1. Why is the hidden nonlinearity scientifically necessary here?**
Without a nonlinear activation function, a sequence of affine transformations collapses mathematically into a single affine transformation. Since the XOR problem is not linearly separable, a single affine transformation cannot solve it. The nonlinearity allows the network to learn a new representation of the inputs where the classes become linearly separable for the final output layer.

**2. Why is sigmoid plus binary cross-entropy a sensible engineering pairing for the output?**
The sigmoid function outputs a value between 0 and 1, which represents the probability of the positive class. Binary cross-entropy is the mathematically correct loss function for binary classification probabilities (derived from maximum likelihood estimation), heavily penalizing confident wrong predictions and providing strong gradients when the prediction is far from the target.

**3. Validation Criteria (Successful Learning Check):**
1.  **Final loss**: Should be close to 0.
2.  **Predicted labels**: All four examples must be classified correctly (predictions > 0.5 for class 1, < 0.5 for class 0).
3.  **Gradient check**: The gradient of the loss with respect to parameters should be non-zero during training and approach zero as it converges.

---

## 3. LLM Prompts and Corrections

**Prompt Used:**
```
Generate minimal PyTorch code for the following model and dataset. Do not change the architecture or task. After training, report the final loss, all four probabilities, thresholded labels, and one parameter-gradient tensor. Set a random seed for reproducibility and explain each test in one sentence.

Dataset: XOR problem (inputs: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0).
Architecture: 2-2-1 network.
Hidden activation: Sigmoid.
Output: Logits with BCEWithLogitsLoss.
Initialisation: Random weights.
Training: Full-batch training for a few thousand lightweight CPU steps.
Print: Final loss, four predictions, and at least one gradient tensor after backward().
```

---

## 4. Final Code 

The final implementation of the neural model can be found at:
[neural_models.py](file:///c:/Users/Vismay%20VS/Desktop/Artificial%20Intelligence/Lab%20Exercises/4_Neural_Models/neural_models.py)

---

## 5. Experimental Results

### Part A: Basic Learning Check
*   *(Record initial and final loss here)*
*   *(Record final probabilities and thresholded predictions here)*

### Part B: Backpropagation Check
*   *(Record observations on `parameter.grad` here)*

### Part C: Symmetry Experiment
*   *(Record observations on training with zero-initialization here)*

### Part D: Activation Experiment

| Hidden activation | Final loss | 4/4 correct? | Early $\|\nabla_{W^{(1)}}L\|_2$ |
| :--- | :--- | :--- | :--- |
| Sigmoid | | | |
| Tanh | | | |
| ReLU | | | |

*Interpretation of results goes here.*

---

## 6. Reflection Questions

1. **What did the XOR experiment demonstrate about the difference between depth and nonlinearity?**
   Depth alone (adding linear layers) does not increase the representational capacity of the model beyond linear functions. Nonlinearity is what allows deeper layers to compute more complex, non-linear decision boundaries.

2. **In your successful run, what evidence showed that backpropagation supplied a useful learning signal rather than merely a nonzero gradient?**
   The fact that the loss decreased consistently over time and the final predictions correctly separated the non-linearly separable XOR inputs showed that the gradients pointed in the direction of a useful internal representation.

3. **Why did identical/zero weight initialisation prevent the two hidden units from learning distinct features?**
   If weights are initialized identically, both hidden units receive the exact same gradients during backpropagation. Consequently, they undergo identical weight updates and will forever compute the same features, effectively reducing the hidden layer to a single unit.

4. **How did changing the hidden activation affect the gradient you observed?**
   *(Answer depends on experimental results - typically ReLU provides stronger, non-vanishing gradients compared to Sigmoid's saturating behavior).*

5. **Why must the output layer and loss be selected together according to the task?**
   The loss function compares the network's output to the target. If the output is a logit, the loss must handle logits (e.g., `BCEWithLogitsLoss`). If the output is a probability, the loss expects probabilities. Mismatching them (e.g., applying MSE to classification or cross-entropy to unbounded regression outputs) breaks the probabilistic interpretation and the gradient dynamics.

6. **Give one example where the LLM improved your engineering productivity and one example where human verification was essential.**
   The LLM improved productivity by rapidly writing the PyTorch training loop and tensor setup boilerplate. Human verification was essential to ensure the LLM correctly implemented `BCEWithLogitsLoss` without adding a redundant sigmoid layer, which is a common hallucination.

7. **Which tests in this laboratory would you keep if the model were scaled up, and which would become too expensive?**
   Tracking training loss, evaluating validation metrics, and checking for vanishing/exploding gradients (e.g., gradient norms) should be kept. Exhaustive symmetry checks on all weights or finite-difference gradient checks would become computationally prohibitive for large models.
