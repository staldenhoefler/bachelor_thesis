# Coding Guidelines for Bachelor Thesis

This document outlines the coding conventions and rules for this project. Adhering to these guidelines ensures consistency, readability, and maintainability of the codebase.

## 1. General Principles

* **Language**:
    * All code-related elements (variable names, function names, class names, comments, docstrings, etc.) **must be in English**.
    * Explanatory texts in Jupyter Notebooks (`.ipynb`) that describe the process, steps, or results **should be in German**.
* **Consistency**: The style should be consistent across the entire project.
* **Clarity**: Code should be written to be as clear and understandable as possible.

---

## 2. Naming Conventions (PEP 8)

We will follow the [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/) for Python code.

### Variables, Functions, and Methods
Use **`snake_case`**: all lowercase with underscores separating words.

* **Example (Variables)**:
    ```python
    learning_rate = 0.01
    user_data = load_data('users.csv')
    ```

* **Example (Functions/Methods)**:
    ```python
    def calculate_mean(data_points):
        # function implementation
        pass

    class DataProcessor:
        def clean_data(self, raw_data):
            # method implementation
            pass
    ```

### Classes
Use **`PascalCase`** (also known as `CapWords`): capitalize the first letter of each word, with no underscores.

* **Example**:
    ```python
    class DataVisualizer:
        pass

    class MachineLearningModel:
        pass
    ```

### Constants
Use **`UPPERCASE_SNAKE_CASE`**: all uppercase with underscores separating words. Constants are variables whose values are not intended to change.

* **Example**:
    ```python
    MAX_ITERATIONS = 1000
    DEFAULT_MODEL_PATH = './models/final_model.pkl'
    ```

### Modules and Packages
Use **`short_lowercase_names`**. Underscores can be used if it improves readability.

* **Example**:
    * `data_preprocessing.py`
    * `utils.py`

---

## 3. Comments and Docstrings

### Comments
Use inline comments to explain complex or non-obvious parts of the code. Comments must be in **English**.

* **Example**:
    ```python
    # Drop rows with missing values to ensure data quality
    cleaned_data = raw_data.dropna()
    ```

### Docstrings
Every public module, function, class, and method should have a docstring. Follow the [PEP 257 docstring conventions](https://www.python.org/dev/peps/pep-0257/). Docstrings must be in **English**.

Use the `Google Style` for docstrings as it is readable and easy to parse.

* **Example (Function)**:
    ```python
    def load_dataset(path: str) -> pd.DataFrame:
        """Loads a dataset from a specified CSV file.

        Args:
            path (str): The file path to the CSV file.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        # implementation
        pass
    ```

* **Example (Class)**:
    ```python
    class ModelTrainer:
        """A class to train a machine learning model.

        Attributes:
            model: The machine learning model instance.
            learning_rate (float): The learning rate for the training process.
        """
        def __init__(self, model, learning_rate=0.01):
            self.model = model
            self.learning_rate = learning_rate
    ```

---

## 4. Git and Commit Messages

All commit messages **must be in English**. We will follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

### Commit Message Structure
&lt;type>[optional scope]: &lt;description>

[optional body]

[optional footer]

* **Common Types**:
    * **`feat`**: A new feature.
    * **`fix`**: A bug fix.
    * **`docs`**: Documentation only changes.
    * **`style`**: Changes that do not affect the meaning of the code (white-space, formatting, etc.).
    * **`refactor`**: A code change that neither fixes a bug nor adds a feature.
    * **`test`**: Adding missing tests or correcting existing tests.
    * **`chore`**: Changes to the build process or auxiliary tools and libraries.

### Commit Message Examples

* **Simple Commit**:
    ```
    feat: Add data visualization module
    ```

* **Commit with Body**:
    ```
    fix: Correct calculation of validation accuracy

    The previous implementation divided by the total number of samples
    instead of the number of validation samples, leading to incorrect
    accuracy scores. This has been resolved.
    ```

* **Commit with Scope**:
    ```
    docs(readme): Update setup instructions for new dependencies
    ```

---

## 5. Jupyter Notebooks (`.ipynb`)

Jupyter Notebooks are used for exploration, analysis, and visualization.

* **Code Cells**: All code within Jupyter Notebook cells must follow the English naming conventions and commenting rules defined above.
* **Markdown Cells**: The explanatory text, descriptions, headings, and interpretations in Markdown cells **should be in German**. This is where you explain your thought process, the steps being taken, and the results of your analysis for the thesis.

### Example of a Jupyter Notebook Cell

**Markdown-Cell (German)**:
> ### 1. Daten laden und vorbereiten
> In diesem Schritt werden die Rohdaten aus der CSV-Datei geladen. Anschließend werden die Daten bereinigt, indem fehlende Werte entfernt und die Datentypen korrigiert werden.

**Code-cell (English)**:
```python
# data_loading.py
import pandas as pd

# Path to the raw data file
raw_data_path = './data/raw/sensor_data.csv'

# Load the dataset using pandas
raw_dataset = pd.read_csv(raw_data_path)

# Display the first few rows to inspect the data
print(raw_dataset.head())

# --- Data Cleaning ---

# Create a copy to avoid modifying the original dataframe
processed_data = raw_dataset.copy()

# Remove rows with any missing values
processed_data.dropna(inplace=True)

print(f"Dataset shape after removing NaNs: {processed_data.shape}")