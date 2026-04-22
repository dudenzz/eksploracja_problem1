
from RatingSystem import RatingSystem

from tqdm import tqdm
import pickle
from surprise import SVDpp, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pandas as pd
from collections import defaultdict

class MySystem(RatingSystem):
    def __init__(self, n_factors: int = 50, n_epochs: int = 20,
                 lr_all: float = 0.007, reg_all: float = 0.02):
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all

        self._movie_avg: dict[int, float] = {}
        self._user_avg: dict[int, float] = {}
        self._global_avg: float = 3.5
        self.testset = None
        self._load_model()
        if self.model is None:
            self._prepare_and_train()

    def _prepare_and_train(self) -> None:
        print("[SVD++ 156145_155941_155260] Przygotowanie danych (Train/Test Split 90/10)...")

        rows = []
        for u_id, user_obj in tqdm(self.users.items(), desc="Ekstrakcja ocen"):
            for m_id, rating in user_obj.ratings.items():
                rows.append((int(u_id), int(m_id), float(rating)))

        df = pd.DataFrame(rows, columns=["userID", "movieID", "rating"])
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df[["userID", "movieID", "rating"]], reader)

        # WYMAGANA ZMIANA: Podział danych zamiast build_full_trainset()
        trainset, testset = train_test_split(data, test_size=0.1, random_state=42)
        self.testset = testset

        # Obliczanie średnich tylko na zbiorze treningowym dla fallbacku
        total_sum = 0.0
        total_count = 0
        movie_sums = defaultdict(list)
        user_sums = defaultdict(list)

        for (u, m, r) in trainset.all_ratings():
            real_u = trainset.to_raw_uid(u)
            real_m = trainset.to_raw_iid(m)
            movie_sums[real_m].append(r)
            user_sums[real_u].append(r)
            total_sum += r
            total_count += 1

        if total_count > 0:
            self._global_avg = total_sum / total_count
        self._movie_avg = {m: sum(v) / len(v) for m, v in movie_sums.items()}
        self._user_avg = {u: sum(v) / len(v) for u, v in user_sums.items()}

        print(f"[SVD++ 156145_155941_155260] Trening na {trainset.n_ratings:,} ocenach...")
        self.model = SVDpp(
            n_factors=self.n_factors, n_epochs=self.n_epochs,
            lr_all=self.lr_all, reg_all=self.reg_all, verbose=False
        )
        self.model.fit(trainset)

        # Ewaluacja na odłożonym zbiorze
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        print(f"[SVD++ 156145_155941_155260] Gotowe! Test RMSE: {rmse:.4f}")

    def rate(self, user, movie) -> float:
        try:
            user_id = user.id if hasattr(user, "id") else int(user)
            movie_id = movie if isinstance(movie, int) else int(movie)
            pred = self.model.predict(user_id, movie_id)
            if pred.details.get("was_impossible", False):
                return self._fallback(user_id, movie_id)
            return float(pred.est)
        except:
            return self._global_avg

    def _load_model(self):
        try:
            with open("156145_155941_155260.pkl", "rb") as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            self.model = None

    def _fallback(self, user_id: int, movie_id: int) -> float:
        if movie_id in self._movie_avg: return self._movie_avg[movie_id]
        if user_id in self._user_avg: return self._user_avg[user_id]
        return self._global_avg

    def __str__(self) -> str:
        return "SVD++ 156145_155941_155260 (CV-Ready)"