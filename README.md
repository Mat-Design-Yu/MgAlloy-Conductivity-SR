# Prediction Workflow for Thermal and Electrical Conductivity in As-cast Magnesium Alloys

This repository provides the official implementation of the prediction workflow associated with our manuscript:

**"Analytical Equations for Thermal and Electrical Conductivity Prediction in As-cast Magnesium Alloys: A Symbolic Regression Approach"**
(Submitted to *Journal of Magnesium and Alloys*)

The primary purpose of this repository is to enable the reproduction and validation of our final predictive models. It contains the necessary code and instructions to predict the thermal and electrical conductivity for new as-cast magnesium alloy compositions using the analytical equations derived in our study.

---

## 🌐 Online Predictive Tool

For convenience and immediate use without any local installation, we have deployed the final predictive models as a user-friendly web application. This tool provides direct access to the predictive capabilities of our work.

**Access the tool here: [https://www.matdesign.cloud/Mg-Conductivity]**

---

## ⚙️ Local Installation and Setup

For researchers who wish to run the prediction workflow locally, we recommend using the `conda` package manager to create a controlled and reproducible environment. This ensures that the underlying feature calculation, which our model depends on, is consistent with our research.

### Prerequisites

*   An installation of `Anaconda` or `Miniconda`.
*   `git` installed on your system.

### Installation Steps

1.  **Create and activate the conda environment:**
    This command creates a dedicated environment named `MGTCECSR_ENV` with Python 3.11.

    ```bash
    conda create -n MGTCECSR_ENV python=3.11
    conda activate MGTCECSR_ENV
    ```

2.  **Install required Python packages:**
    These packages are necessary for data handling, scientific computing, and running the Jupyter Notebook.

    ```bash
    pip install chardet jupyter notebook scikit-learn seaborn ipympl openpyxl tqdm
    ```

3.  **Install specific forks of materials science libraries:**
    To guarantee absolute reproducibility of the feature generation process, our workflow relies on specific versions of `pymatgen` and `matminer`. We have forked these libraries to preserve the exact state used in our manuscript. Please install them from our GitHub repositories.

    ```bash
    # Ensure pip is up-to-date within the environment
    python -m pip install --upgrade pip

    # Create a directory for these custom packages
    mkdir packages && cd packages

    # Clone and install our fork of matminer for consistent feature calculation
    git clone -b main_for_yu https://github.com/Mat-Design-Yu/matminer_for_yu.git
    cd matminer_for_yu
    pip install -e .
    cd ..

    # Clone and install our fork of pymatgen, a dependency for matminer
    git clone -b master_for_yu https://github.com/Mat-Design-Yu/pymatgen_for_yu.git
    cd pymatgen_for_yu
    pip install -e .
    cd ..
    ```

---

## 💻 How to Use the Prediction Script

After successfully setting up the environment, you can use the `thermal_conductivity_calculator.py` script to predict properties for any alloy of interest.

1.  **Modify the Input in the Script:**
    *   Open the `thermal_conductivity_calculator.py` file in a text editor.
    *   Locate the main execution block at the bottom of the file (`if __name__ == "__main__":`).
    *   Change the `test_composition` dictionary and `test_temperature` variable to your desired values. The composition must be a Python dictionary of element symbols and their corresponding **atomic percent (at.%)**. The temperature must be in **Kelvin (K)**.

    ```python
    # --- Main program entry point ---
    if __name__ == "__main__":
        # ... (code) ...

        # Define the alloy composition and temperature to be tested
        # MODIFY THE LINES BELOW
        test_composition = {"Mg": 96, "Al": 1, "Zn": 3}
        test_temperature = 298

        # ... (code) ...
    ```

2.  **Run the Prediction Script:**
    *   Ensure your conda environment is active: `conda activate MGTCECSR_ENV`
    *   Navigate to the repository's root directory in your terminal.
    *   Execute the script:
        ```bash
        python thermal_conductivity_calculator.py
        ```

3.  **Interpret the Console Output:**
    *   The script will print the results directly to your terminal. The output will include:
        *   The input composition and temperature you defined.
        *   The predicted thermal and electrical conductivity values.

---

## 📄 License

The code in this repository is released under the GPL-3.0 License. Please see the `LICENSE` file for more details.

---

## 📧 Contact

For questions or inquiries regarding this work, please contact the corresponding author:

*   **Professor Zhigang Yu** - `yuzg126@126.com`