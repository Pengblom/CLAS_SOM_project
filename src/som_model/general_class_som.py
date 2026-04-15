import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pickle


class GeneralCLASSOM:

    def __init__(
        self,
        x=20,
        y=20,
        input_len=6,
        random_seed=42,

        **som_kwargs
    ):

        self.x = x
        self.y = y
        self.input_len = input_len
        self.random_seed = random_seed

        # Default parameters (can be overridden)
        default_params = dict(
            sigma=1.0,
            learning_rate=0.5,
            decay_function="asymptotic_decay",
            neighborhood_function="gaussian",
            topology="rectangular",
            activation_distance="euclidean",
            sigma_decay_function="asymptotic_decay",
            random_seed=random_seed
        )

        # merge defaults + user params
        self.som_params = {**default_params, **som_kwargs}

        # documentation
        self.params = {
            "x": x,
            "y": y,
            "input_len": input_len,
            **self.som_params
        }

        # initialize MiniSom
        self.som = MiniSom(
            x=x,
            y=y,
            input_len=input_len,
            **self.som_params
        )

        self.scaler = MinMaxScaler()
        self.kmeans = None
        self.cluster_method = None
        self.is_trained = False
        self.cluster_labels = {}

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    def train(
        self,
        data,
        iterations=10000,
        mode="batch",
        use_pca=True,
        verbose=True,
        use_epochs=False
    ):

        scaled_data = self.scaler.fit_transform(data)

        # weight initialization
        if use_pca:
            self.som.pca_weights_init(scaled_data)
        else:
            self.som.random_weights_init(scaled_data)

        # training modes
        if mode == "batch":
            self.som.train_batch(scaled_data, iterations, verbose=verbose)

        elif mode == "random":
            self.som.train_random(scaled_data, iterations, verbose=verbose)

        elif mode == "standard":
            self.som.train(
                scaled_data,
                iterations,
                verbose=verbose,
                use_epochs=use_epochs
            )

        else:
            raise ValueError("mode must be: batch | random | standard")

        self.is_trained = True
        print(f"Training completed ({iterations} iterations)")

    # --------------------------------------------------
    # CLUSTERING
    # --------------------------------------------------

    def create_clusters(
        self,
        data=None,
        n_clusters=3,
        method="standard"
    ):

        if not self.is_trained:
            raise RuntimeError("Train model first.")

        weights = self.som.get_weights().reshape(self.x * self.y, -1)

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_seed,
            n_init=10
        )

        # -------------------------
        # STANDARD CLUSTERING
        # -------------------------
        if method == "standard":

            labels = self.kmeans.fit_predict(weights)

        # -------------------------
        # DENSITY-AWARE CLUSTERING
        # -------------------------
        elif method == "density":

            if data is None:
                raise ValueError(
                    "Density method requires original data."
                )

            scaled_data = self.scaler.transform(data)

            # neuron usage frequency
            activation = self.som.activation_response(scaled_data)
            density = activation.flatten()

            # weighted KMeans
            self.kmeans.fit(weights, sample_weight=density)
            labels = self.kmeans.labels_

        # NEURON CLUSTERING
        elif method == "neurons":
            self.kmeans.fit(weights)
            labels = self.kmeans.labels_

        else:
            raise ValueError(
                "method must be 'standard' or 'density' or 'neurons'"
            )

        self.cluster_method = method

        return labels.reshape(self.x, self.y)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    def predict_cluster(self, data):

        if self.kmeans is None:
            raise RuntimeError("Run create_clusters() first.")

        scaled_data = self.scaler.transform(data)

        winners = np.array([self.som.winner(d) for d in scaled_data])
        flat_idx = winners[:, 0] * self.y + winners[:, 1]

        return self.kmeans.labels_[flat_idx]

    # --------------------------------------------------
    # CLUSTER LABELING
    # --------------------------------------------------

    def set_cluster_labels(self, label_dict):
        self.cluster_labels = label_dict

    def get_cluster_name(self, cluster_id):
        return self.cluster_labels.get(cluster_id, "Unknown")

    # --------------------------------------------------
    # UTILITIES
    # --------------------------------------------------

    def get_u_matrix(self):
        return self.som.distance_map()

    def get_summary(self):
        return self.params

    # --------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------

    def save_model(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
            
    def export_to_c(self, filepath="som_model.h"):

        if not self.is_trained:
            raise RuntimeError("Model must be trained before export.")

        if self.kmeans is None:
            raise RuntimeError("Clusters must be created before export.")

        weights = self.som.get_weights()
        min_vals = self.scaler.data_min_
        range_vals = self.scaler.data_range_
        range_vals = np.where(range_vals == 0, 1e-6, range_vals)

        x, y, input_len = weights.shape
        neurons = x * y

        weights_2d = weights.reshape(neurons, input_len)

        # cluster per neuron (inte per sample)
        clusters = self.kmeans.predict(weights_2d)

        with open(filepath, "w") as f:

            f.write("#pragma once\n\n")

            f.write(f"#define SOM_NEURONS {neurons}\n")
            f.write(f"#define SOM_INPUT_LEN {input_len}\n")
            f.write(f"#define SOM_WIDTH {x}\n")
            f.write(f"#define SOM_HEIGHT {y}\n\n")

            # -----------------------
            # SOM WEIGHTS
            # -----------------------
            f.write("static const float som_weights[] = {\n")

            flat = weights_2d.flatten()

            for i, w in enumerate(flat):
                if i % 6 == 0:
                    f.write("   ")
                f.write(f"{w:.8f}f,")
                if i % 6 == 5:
                    f.write("\n")

            f.write("};\n\n")

            # -----------------------
            # SCALER MIN
            # -----------------------
            f.write("static const float scaler_min[] = {\n")
            for v in min_vals:
                f.write(f"   {v:.8f}f,\n")

            f.write("};\n\n")

            # -----------------------
            # SCALER RANGE
            # -----------------------
            f.write("static const float scaler_range[] = {\n")
            for v in range_vals:
                f.write(f"   {v:.8f}f,\n")

            f.write("};\n\n")

            # -----------------------
            # CLUSTERS (per neuron)
            # -----------------------
            f.write("static const uint8_t som_clusters[] = {\n")

            for i, c in enumerate(clusters):

                if i % 16 == 0:
                    f.write("   ")

                f.write(f"{int(c)},")

                if i % 16 == 15:
                    f.write("\n")

            f.write("\n};\n")
            # -----------------------
            # CLUSTER NAMES
            # -----------------------
            if hasattr(self, "cluster_labels") and self.cluster_labels:

                f.write("\n// Cluster names\n")
                f.write("static const char* cluster_names[] = {\n")

            # säkerställ rätt ordning (0,1,2,...)
                for i in range(len(self.cluster_labels)):
                    name = self.cluster_labels[i]
                    f.write(f'   "{name}",\n')

                f.write("};\n")

        print(f"C header exported to: {filepath}")
        