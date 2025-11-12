## 🎓 Mathematical Concepts of Linear Algebra and Optimization Using a Generative Artificial Intelligence Tool – Implementation of the Steepest Descent Method

This project was presented at the **1st Student and Mathematical Conference** of the University of the Peloponnese 🏫, organized by the Department of Digital Systems 💻 on *March 14, 2025*. Its aim was to bridge theoretical mathematical concepts with modern artificial intelligence technologies through the practical implementation of the Steepest Descent algorithm.

### 🎯 Project Objective
The purpose of this work is to:

- Highlight the connection between *linear algebra 📚* , *optimization 📈* , and *artificial intelligence 🤖* .

- Demonstrate how the **Steepest Descent method** can be used to find the minimum of a function with two variables.

- Showcase how a **Generative AI Tool** can support the educational process by automatically generating assessment material and enhancing conceptual understanding.

### 🧐 Code Overview 

#### 1️⃣ Steepest Descent Algorithm Implementation (Steepest_Descent.py)

The first code implements the Steepest Descent algorithm to minimize a two-variable function. The user is prompted to input:

- Initial starting points: $(x_0, y_0)$,

- Learning rate: $a$,

- Three termination criteria: $c_1, c_2, c_3$,


🧮 The algorithm calculates partial derivatives and the gradient, updating the coordinates at each iteration. The process terminates when:

- The gradient norm is smaller than $𝑐_1$

- The distance between successive points is less than $𝑐_2$

- The difference in function values between iterations is less than $𝑐_3$

- Or when the maximum number of iterations (1000) is exceeded


🙍🏻‍♂️ The user receives:

- A visual representation of the algorithm’s path in both 2D and 3D graphs

- Final values of the variables and the function

- A message indicating which termination criterion was satisfied



#### 2️⃣ Question Generation Based on Algorithm Parameters (API_Conf_DS_2025.py)
The second code uses the OpenAI API to generate multiple-choice questions based on the algorithm’s logic and structure 🧠, without external documents. The questions are created in Greek and are tailored to the Steepest Descent method, using a predefined prompt that includes theoretical context and algorithmic details. The difficulty of each question is defined using a Likert scale (1–5), and the output is saved in a text file.

#### 3️⃣ Question Generation Based on PDF Input (API_PDF.py)
The third code also uses the OpenAI API, but with a different approach: it enhances the language model’s input by providing a PDF document 📝 containing theoretical or practical content related to the Steepest Descent method. The model extracts relevant information from the PDF and generates questions accordingly, allowing for deeper contextualization and more accurate alignment with the source material.

#### 🔍 Key Difference Between Code 2 and Code 3
Code 2: Generates questions based on predefined algorithmic data and prompts, focusing on the internal logic of the Steepest Descent method.

Code 3: Enhances the model’s understanding by feeding it a PDF document, from which it extracts information to generate questions. This allows for more nuanced and content-rich question creation, especially useful when working with educational or scientific texts.

### 📊 What the Code Demonstrates to the User
- The practical application of optimization theory

- The dynamic path toward a function’s minimum

- Interactive control over algorithm parameters and visualization

- The integration of artificial intelligence in educational workflows

- A clear connection between mathematical theory and modern AI tools, enhancing student engagement and understanding
