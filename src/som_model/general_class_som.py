import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import RobustScaler
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

        self.scaler = RobustScaler()
        self.kmeans = None
        self.cluster_method = None
        self.is_trained = False

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

        else:
            raise ValueError("method must be 'standard' or 'density'")

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