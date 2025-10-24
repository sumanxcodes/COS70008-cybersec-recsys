# funk_svd.py
import numpy as np
from scipy.sparse import csr_matrix

class FunkSVD:
    """
    FunkSVD with user/item biases trained by SGD on observed entries of a CSR matrix.

    Params
    ------
    n_factors : int
        Latent dimensionality.
    n_epochs : int
        Max epochs for SGD.
    lr : float
        Learning rate for all parameters.
    reg_bias : float
        L2 regularization for biases (bu, bi).
    reg_factors : float
        L2 regularization for factor matrices (U, V).
    seed : int
        RNG seed.
    clip : tuple[float, float] | None
        If set, predictions are clipped to (min_rating, max_rating).
    shuffle : bool
        Shuffle training triplets each epoch.
    verbose : bool
        Print RMSE every eval.
    early_stopping : bool
        If True and R_val is provided, keep best params by val RMSE with patience.
    patience : int
        Stop if val RMSE doesn't improve for this many eval steps.
    eval_every : int
        Compute metrics every N epochs.
    dtype : np.dtype
        Float dtype for model parameters.
    """

    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 30,
        lr: float = 0.01,
        reg_bias: float = 0.01,
        reg_factors: float = 0.02,
        seed: int = 42,
        clip: tuple | None = None,
        shuffle: bool = True,
        verbose: bool = True,
        early_stopping: bool = True,
        patience: int = 3,
        eval_every: int = 1,
        dtype=np.float32,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg_bias = reg_bias
        self.reg_factors = reg_factors
        self.rng = np.random.default_rng(seed)
        self.clip = clip
        self.shuffle = shuffle
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.patience = patience
        self.eval_every = max(1, eval_every)
        self.dtype = dtype

        # learned params
        self.mu = None          # global mean
        self.bu = None          # user/org biases
        self.bi = None          # item/exercise biases
        self.U = None           # user/org factors (n_users, k)
        self.V = None           # item/exercise factors (n_items, k)

        # training history
        self.history_ = {"epoch": [], "rmse": [], "val_rmse": []}

    # --------- helpers ---------

    @staticmethod
    def _to_csr(R):
        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)
        R.sort_indices()  # makes row slicing + seen-item lookup cheap
        return R

    @staticmethod
    def _iter_known(R: csr_matrix):
        # iterate over non-zeros
        Rcoo = R.tocoo()
        for u, i, r in zip(Rcoo.row, Rcoo.col, Rcoo.data):
            yield int(u), int(i), float(r)

    def _predict_raw(self, u: int, i: int):
        return self.mu + self.bu[u] + self.bi[i] + self.U[u].dot(self.V[i])

    def _predict_clipped(self, val: float):
        if self.clip is None:
            return val
        lo, hi = self.clip
        return float(np.clip(val, lo, hi))

    # --------- API ---------

    def fit(self, R: csr_matrix, R_val: csr_matrix | None = None):
        """Train on known entries in R (users×items CSR)."""
        R = self._to_csr(R)
        if R_val is not None:
            R_val = self._to_csr(R_val)

        n_users, n_items = R.shape

        # init params
        self.mu = float(R.data.mean()) if R.nnz > 0 else 0.0
        self.bu = np.zeros(n_users, dtype=self.dtype)
        self.bi = np.zeros(n_items, dtype=self.dtype)
        self.U = 0.1 * self.rng.standard_normal((n_users, self.n_factors)).astype(self.dtype)
        self.V = 0.1 * self.rng.standard_normal((n_items, self.n_factors)).astype(self.dtype)

        # training triplets
        Rcoo = R.tocoo()
        users = Rcoo.row.astype(np.int64)
        items = Rcoo.col.astype(np.int64)
        rates = Rcoo.data.astype(self.dtype)
        n_obs = len(rates)

        best = {
            "rmse": np.inf,
            "val_rmse": np.inf,
            "epoch": 0,
            "mu": None, "bu": None, "bi": None, "U": None, "V": None,
        }
        no_improve = 0
        self.history_ = {"epoch": [], "rmse": [], "val_rmse": []}

        for ep in range(1, self.n_epochs + 1):
            # shuffle indices each epoch if requested
            if self.shuffle:
                order = self.rng.permutation(n_obs)
            else:
                order = np.arange(n_obs, dtype=np.int64)

            # SGD over all observed entries
            for idx in order:
                u = users[idx]; i = items[idx]; r = rates[idx]

                # prediction + error
                pred = self.mu + self.bu[u] + self.bi[i] + self.U[u].dot(self.V[i])
                err = r - pred

                # update biases
                self.bu[u] += self.lr * (err - self.reg_bias * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg_bias * self.bi[i])

                # update factors (use copies to avoid aliasing bug)
                u_old = self.U[u].copy()
                i_old = self.V[i].copy()
                self.U[u] += self.lr * (err * i_old - self.reg_factors * u_old)
                self.V[i] += self.lr * (err * u_old - self.reg_factors * i_old)

            # eval
            if ep % self.eval_every == 0:
                train_rmse = self.rmse(R)
                val_rmse = None
                if R_val is not None:
                    val_rmse = self.rmse(R_val)

                self.history_["epoch"].append(ep)
                self.history_["rmse"].append(train_rmse)
                self.history_["val_rmse"].append(val_rmse)

                if self.verbose:
                    msg = f"[epoch {ep:>3}] train RMSE={train_rmse:.4f}"
                    if val_rmse is not None:
                        msg += f" | val RMSE={val_rmse:.4f}"
                    print(msg)

                # early stopping (track best by val if provided, else by train)
                metric = val_rmse if R_val is not None else train_rmse
                if metric is not None and metric + 1e-8 < (best["val_rmse"] if R_val is not None else best["rmse"]):
                    best.update({
                        "rmse": train_rmse,
                        "val_rmse": val_rmse if val_rmse is not None else best["val_rmse"],
                        "epoch": ep,
                        "mu": self.mu,
                        "bu": self.bu.copy(),
                        "bi": self.bi.copy(),
                        "U": self.U.copy(),
                        "V": self.V.copy(),
                    })
                    no_improve = 0
                else:
                    no_improve += 1
                    if self.early_stopping and R_val is not None and no_improve >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {ep} (best epoch {best['epoch']})")
                        break  # out of epoch loop

            # stop outer loop too if early stopped
            if self.early_stopping and R_val is not None and no_improve >= self.patience:
                break

        # restore best params
        if best["epoch"] > 0:
            self.mu = best["mu"]
            self.bu = best["bu"]
            self.bi = best["bi"]
            self.U = best["U"]
            self.V = best["V"]

        # ✅ FIX: Return training history for Streamlit plots
        history = list(
            zip(
                self.history_["epoch"],
                self.history_["rmse"],
                self.history_["val_rmse"]
            )
        )
        return history


    def predict_one(self, u: int, i: int) -> float:
        """Predict rating for user/org u and item/exercise i."""
        val = self._predict_raw(u, i)
        return self._predict_clipped(val)

    def predict_for_org(self, u: int) -> np.ndarray:
        """Predict ratings for all items for org u (vector)."""
        scores = self.mu + self.bu[u] + self.bi + self.U[u] @ self.V.T
        if self.clip is not None:
            lo, hi = self.clip
            scores = np.clip(scores, lo, hi)
        return scores

    def recommend_topn(self, R: csr_matrix, u: int, n: int = 5, exclude_seen: bool = True):
        """
        Recommend top-n item indices for org u.
        If exclude_seen, filter out items with an observed rating in R.
        """
        R = self._to_csr(R)
        scores = self.predict_for_org(u)

        if exclude_seen:
            start, end = R.indptr[u], R.indptr[u + 1]
            seen = set(R.indices[start:end])
        else:
            seen = set()

        # get top-n without materialising all pairs
        # mask seen by setting to -inf
        if seen:
            scores = scores.copy()
            scores[list(seen)] = -np.inf

        # argpartition for speed, then sort the slice
        n = min(n, scores.size - len(seen))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return top.tolist()

    def rmse(self, R: csr_matrix) -> float:
        """RMSE on known ratings in R."""
        se = 0.0
        cnt = 0
        for u, i, r in self._iter_known(R):
            p = self._predict_raw(u, i)
            if self.clip is not None:
                p = self._predict_clipped(p)
            se += (r - p) ** 2
            cnt += 1
        return float(np.sqrt(se / max(cnt, 1)))


# -------------------------
# Minimal usage example
# -------------------------
if __name__ == "__main__":
    # tiny demo with random observed entries
    rng = np.random.default_rng(0)
    n_users, n_items = 50, 100
    nnz = 1000
    rows = rng.integers(0, n_users, size=nnz)
    cols = rng.integers(0, n_items, size=nnz)
    vals = rng.normal(5.0, 2.0, size=nnz).clip(0, 10)

    R = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

    # simple train/val split on entries
    mask = rng.random(nnz) < 0.8
    R_train = csr_matrix((vals[mask], (rows[mask], cols[mask])), shape=R.shape)
    R_val   = csr_matrix((vals[~mask], (rows[~mask], cols[~mask])), shape=R.shape)

    model = FunkSVD(
        n_factors=32,
        n_epochs=50,
        lr=0.01,
        reg_bias=0.01,
        reg_factors=0.05,
        seed=42,
        clip=(0.0, 10.0),
        verbose=True,
        early_stopping=True,
        patience=3,
        eval_every=1,
    ).fit(R_train, R_val)

    print("Top-5 for user 0:", model.recommend_topn(R_train, u=0, n=5))
