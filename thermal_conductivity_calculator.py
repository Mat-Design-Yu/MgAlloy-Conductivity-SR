# thermal_conductivity_calculator.py

import sys
import os
import pandas as pd
import numpy as np
import warnings

# --- Import necessary libraries ---
from packages.pymatgen_for_yu.pymatgen.core import Composition
from packages.matminer_for_yu.matminer.featurizers.composition.composite import (
    ElementProperty,
)
from packages.matminer_for_yu.matminer.featurizers.composition.alloy import WenAlloys
from multiprocessing import freeze_support

# --- Global constants and configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SR_EQUATION_PATH = os.path.join(CURRENT_DIR, "assets", "sr_equations.csv")

FEATURE_ORDER = [
    "T [K]",
    "MagpieData std_dev NValence",
    "Radii gamma",
    "Mean cohesive energy",
]

EC_FORMULA_MULTIPLIER = 2.2054725000682553e-08
EC_FORMULA_OFFSET = 4.359468758336787


class ThermalConductivityPredictor:
    """
    A calculator for predicting magnesium alloy properties and finding similar experimental data.
    The output list of similar alloys is 1-indexed, and the index column has a title.
    """

    TEMP_SCALE_FACTOR = 100.0

    SIMILARITY_NORMALIZATION_FACTOR = 15.0

    def __init__(self):
        """
        Constructor, initializes the predictor and loads data.
        """
        print("Initializing ThermalConductivityPredictor...")
        self.magpie_featurizer = ElementProperty.from_preset("magpie_less")
        self.wen_featurizer = WenAlloys()
        self.predict_function = self._load_and_compile_equation()
        self.experimental_data, self.element_cols = self._load_experimental_data()

    def _load_and_compile_equation(self):
        """
        Loads the symbolic regression equation from a CSV file and compiles it into a callable function.
        """
        try:
            equations_df = pd.read_csv(SR_EQUATION_PATH)
            equation_row = equations_df.iloc[0]
            equation_body = equation_row["Equation"]

            # Replace common math function names with their numpy equivalents
            import re

            replacements = {
                "sqrt": "np.sqrt",
                "log": "np.log",
                "exp": "np.exp",
                "square": "np.square",
                "pow": "np.power",
                "cos": "np.cos",
                "sin": "np.sin",
                "tan": "np.tan",
            }
            for key, value in replacements.items():
                equation_body = re.sub(r"\b" + key + r"\b", value, equation_body)

            # Replace feature placeholders (x0, x1, ...) with numpy array indexing
            for i in range(len(FEATURE_ORDER)):
                equation_body = equation_body.replace(f"x{i}", f"X[:, {i}]")

            # Create a lambda function from the equation string
            lambda_str = f"lambda X: {equation_body}"
            predict_func = eval(lambda_str)

            print(
                f"Successfully loaded and compiled equation: {equation_row['Equation']}"
            )
            return predict_func
        except Exception as e:
            print(f"FATAL ERROR during model initialization: {e}")
            return None

    def _load_experimental_data(self):
        """
        Loads and preprocesses the experimental data from an Excel file.
        """
        try:
            EXPERIMENTAL_DATA_PATH = os.path.join(CURRENT_DIR, "data", "data.xlsx")
            df = pd.read_excel(EXPERIMENTAL_DATA_PATH)
            print(
                f"Successfully loaded experimental data from '{EXPERIMENTAL_DATA_PATH}'."
            )

            # Define the expected element composition columns
            element_cols = [
                "Mg [at.%]",
                "Al [at.%]",
                "Zn [at.%]",
                "Sn [at.%]",
                "Mn [at.%]",
                "Zr [at.%]",
                "Ca [at.%]",
                "Y [at.%]",
                "Cu [at.%]",
                "Ce [at.%]",
                "Sm [at.%]",
                "Si [at.%]",
                "La [at.%]",
                "Sr [at.%]",
                "Pb [at.%]",
                "Gd [at.%]",
                "Ag [at.%]",
                "Nd [at.%]",
                "Sc [at.%]",
                "Li [at.%]",
            ]

            # Ensure all element columns exist, filling missing ones with 0
            for col in element_cols:
                if col not in df.columns:
                    df[col] = 0

            df[element_cols] = df[element_cols].fillna(0)
            df[element_cols] = df[element_cols].astype(float)
            df["T [K]"] = pd.to_numeric(df["T [K]"], errors="coerce")
            df.dropna(subset=["T [K]"], inplace=True)

            return df, element_cols
        except FileNotFoundError:
            print(
                f"WARNING: Experimental data file not found at '{EXPERIMENTAL_DATA_PATH}'."
            )
            return None, None
        except Exception as e:
            print(f"ERROR loading experimental data: {e}")
            return None, None

    def _calculate_features(self, composition: Composition) -> dict:
        """
        Calculates the required material features for a given composition.
        """
        comp_df = pd.DataFrame([{"composition": composition}])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            df_magpie = self.magpie_featurizer.featurize_dataframe(
                comp_df, "composition", ignore_errors=True
            )
            df_wen = self.wen_featurizer.featurize_dataframe(
                comp_df, "composition", ignore_errors=True
            )
            full_features_df = pd.concat(
                [df_magpie.iloc[:, 2:], df_wen.iloc[:, 2:]], axis=1
            )

        features = {}
        try:
            features["MagpieData std_dev NValence"] = full_features_df[
                "MagpieData std_dev NValence"
            ].iloc[0]
            features["Radii gamma"] = full_features_df["Radii gamma"].iloc[0]
            features["Mean cohesive energy"] = full_features_df[
                "Mean cohesive energy"
            ].iloc[0]
        except KeyError as e:
            print(f"FATAL ERROR: A required feature could not be found: {e}")
            raise e
        return features

    def predict(self, composition_dict: dict, temperature_k: float) -> dict:
        """
        Predicts thermal and electrical conductivity for a given alloy composition and temperature.
        """
        if not self.predict_function:
            raise RuntimeError("Predictor not initialized properly.")
        try:
            composition = Composition(composition_dict)
        except Exception as e:
            raise ValueError(
                f"Invalid chemical composition: '{composition_dict}'. Error: {e}"
            )

        composition_features = self._calculate_features(composition)

        # Assemble the feature vector in the correct order for the prediction model
        feature_vector = np.array(
            [
                np.sqrt(temperature_k),
                composition_features.get("MagpieData std_dev NValence", 0),
                composition_features.get("Radii gamma", 0),
                composition_features.get("Mean cohesive energy", 0),
            ]
        )

        X_pred = feature_vector.reshape(1, -1)

        # Predict thermal conductivity using the compiled function
        thermal_conductivity = self.predict_function(X_pred).item()
        # Calculate electrical conductivity using the Wiedemann-Franz law approximation
        electrical_conductivity = (
            (thermal_conductivity - EC_FORMULA_OFFSET)
            / temperature_k
            / EC_FORMULA_MULTIPLIER
        )

        return {
            "thermal_conductivity": thermal_conductivity,
            "electrical_conductivity": electrical_conductivity,
        }

    def find_similar_experimental_data(
        self,
        composition_dict: dict,
        temperature_k: float,
        calculation_type: str = "both",
    ) -> pd.DataFrame:
        """
        Finds similar experimental data and adaptively returns a number of results
        based on the count of high-similarity samples.

        Args:
            composition_dict: Dictionary of the alloy composition.
            temperature_k: Temperature in Kelvin.
            calculation_type: The type of calculation ("thermal", "electrical", or "both").
        """
        if self.experimental_data is None or self.element_cols is None:
            print("Warning: Experimental data not loaded. Cannot find similar alloys.")
            return pd.DataFrame()

        # Create a vector for the input composition
        input_comp_vector = np.zeros(len(self.element_cols))
        comp_obj = Composition(composition_dict)
        normalized_comp = comp_obj.fractional_composition.as_dict()

        for i, col_name in enumerate(self.element_cols):
            element_symbol = col_name.split(" ")[0]
            if element_symbol in normalized_comp:
                input_comp_vector[i] = normalized_comp[element_symbol] * 100

        # Extract database compositions and temperatures
        db_compositions = self.experimental_data[self.element_cols].values
        db_temperatures = self.experimental_data["T [K]"].values

        # Calculate Euclidean distance for composition and scaled distance for temperature
        comp_dist_sq = np.sum(np.square(db_compositions - input_comp_vector), axis=1)
        temp_dist_sq = np.square(
            (db_temperatures - temperature_k) / self.TEMP_SCALE_FACTOR
        )
        total_dist = np.sqrt(comp_dist_sq + temp_dist_sq)

        # Convert distance to a similarity score from 0 to 100
        similarity_score = 100 / (1 + total_dist / self.SIMILARITY_NORMALIZATION_FACTOR)

        results_df = self.experimental_data.copy()
        results_df["Similarity [%]"] = similarity_score

        results_df.sort_values(by="Similarity [%]", ascending=False, inplace=True)

        # Adaptively determine how many results to show based on similarity
        high_similarity_count = results_df[results_df["Similarity [%]"] >= 90].shape[0]

        if high_similarity_count >= 11:
            n_to_return = 15
            reason = f"Found {high_similarity_count} (>=11) high similarity alloys"
        elif high_similarity_count >= 6:
            n_to_return = 10
            reason = f"Found {high_similarity_count} (6-10) high similarity alloys"
        else:
            n_to_return = 5
            reason = f"Found {high_similarity_count} (<=5) high similarity alloys"

        print(f"\n--- Similar Alloy Search Information ---")
        print(
            f"Analysis complete: {reason}, the program will display the Top {n_to_return} results."
        )

        # Select output columns based on calculation type
        base_cols = ["Formula [at.%]", "T [K]"]

        if calculation_type == "thermal":
            output_cols = base_cols + ["λ [W·m-1·K-1]"]
        elif calculation_type == "electrical":
            output_cols = base_cols + ["σ [S·m-1]"]
        else:  # "both"
            output_cols = base_cols + ["λ [W·m-1·K-1]", "σ [S·m-1]"]

        n_actual = min(n_to_return, len(results_df))
        top_n_df = results_df.head(n_actual)[output_cols].copy()

        # 1. Ensure the Formula column is not NaN
        if "Formula [at.%]" in top_n_df.columns:

            def format_formula(x):
                return str(x) if pd.notna(x) else "N/A"

            top_n_df["Formula [at.%]"] = top_n_df["Formula [at.%]"].apply(
                format_formula
            )

        # 2. Format temperature as an integer
        if "T [K]" in top_n_df.columns:

            def format_temperature(x):
                if pd.notna(x) and isinstance(x, (int, float)):
                    return str(int(x))  # convert to integer
                return str(x) if pd.notna(x) else "N/A"

            top_n_df["T [K]"] = top_n_df["T [K]"].apply(format_temperature)

        # 3. Handle conductivity columns: display nulls as empty strings
        conductivity_cols = ["λ [W·m-1·K-1]", "σ [S·m-1]"]
        for col in conductivity_cols:
            if col in top_n_df.columns:

                def format_conductivity(x):
                    if pd.notna(x) and isinstance(x, (int, float)):
                        return f"{x:.3f}"
                    return (
                        str(x) if pd.notna(x) else ""
                    )  # display null value as an empty string

                top_n_df[col] = top_n_df[col].apply(format_conductivity)

        # Adjust the index from 0-based to 1-based
        final_df = top_n_df.reset_index(drop=True)
        final_df.index = final_df.index + 1
        final_df.index.name = "No."

        # Add the reason information to the returned DataFrame (as an attribute)
        final_df.attrs["reason"] = reason
        final_df.attrs["n_results"] = n_to_return

        return final_df


# --- Main program entry point ---
if __name__ == "__main__":
    freeze_support()

    predictor = ThermalConductivityPredictor()

    if predictor.predict_function:
        try:
            print("\n" + "=" * 25)
            print("--- Alloy Prediction and Similarity Search ---")
            print("=" * 25)

            # Define the alloy composition and temperature to be tested
            test_composition = {"Mg": 96, "Al": 1, "Zn": 3}
            test_temperature = 298

            # 1. Perform property prediction
            properties = predictor.predict(test_composition, test_temperature)

            print(f"Input Composition: {test_composition}")
            print(f"Input Temperature: {test_temperature} K\n")
            print("--- Prediction Results ---")
            print(
                f"  -> Predicted Thermal Conductivity (λ): {properties['thermal_conductivity']:.3f} W·m-1·K-1"
            )
            print(
                f"  -> Predicted Electrical Conductivity (σ): {properties['electrical_conductivity'] / 1e6:.3f} MS/m"
            )

            # 2. Find similar experimental data
            similar_alloys_df = predictor.find_similar_experimental_data(
                test_composition, test_temperature
            )

            if not similar_alloys_df.empty:
                print(
                    "\n--- Similar Alloys in Experimental Database (Adaptive Results) ---"
                )
                # Set pandas display options for better console output
                pd.set_option("display.width", 120)
                print(similar_alloys_df.to_string())

        except (ValueError, RuntimeError, Exception) as e:
            print(f"\nAn error occurred during execution: {e}")
            import traceback

            traceback.print_exc()
