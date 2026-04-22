import math
import random
import re
from collections import defaultdict
from RatingSystem import RatingSystem
from RatingLib import Movie


class System155864(RatingSystem) :
	def __init__(self, k_user=15, k_item=15, max_comparisons=150) :
		super().__init__()
		self.k_user = k_user
		self.k_item = k_item
		self.max_comparisons = max_comparisons

		self.movie_user_ratings = defaultdict(list)

		# 1. Global Mean
		all_ratings = []
		for ratings in self.movie_ratings.values() :
			all_ratings.extend(ratings)
		self.global_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 2.5

		self.b_i = {}  # Movie Bias
		lambda_i = 10  # Siła przyciągania do średniej globalnej dla filmów
		for mid, ratings in self.movie_ratings.items() :
			self.b_i[mid] = sum(r - self.global_mean for r in ratings) / (len(ratings) + lambda_i)

		self.b_u = {}  # User Bias
		lambda_u = 5  # Siła przyciągania do zera dla użytkowników
		for uid, user_obj in self.users.items() :
			u_ratings = user_obj.ratings
			if not u_ratings :
				self.b_u[uid] = 0.0
				continue

			sum_err = 0.0
			for mid, r in u_ratings.items() :
				bi = self.b_i.get(mid, 0.0)
				sum_err += (r - self.global_mean - bi)

			self.b_u[uid] = sum_err / (len(u_ratings) + lambda_u)

			# Odwrócony indeks
			for mid, rating in u_ratings.items() :
				self.movie_user_ratings[mid].append((uid, rating))

	def _tokenize_title(self, title) :
		clean_title = re.sub(r'\(\d{4}\)', '', title).lower()
		tokens = re.findall(r'\b[a-z]+\b', clean_title)
		stop_words = {'the', 'a', 'an', 'and', 'of', 'in', 'to', 'part', 'vol'}
		return set(t for t in tokens if t not in stop_words)

	def _title_similarity(self, title1, title2) :
		tokens1 = self._tokenize_title(title1)
		tokens2 = self._tokenize_title(title2)
		if not tokens1 or not tokens2 :
			return 0.0
		return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

	def _jaccard_similarity(self, genres1, genres2) :
		if isinstance(genres1, str) :
			genres1 = genres1.split('|')
		if isinstance(genres2, str) :
			genres2 = genres2.split('|')

		set1 = set(genres1)
		set2 = set(genres2)

		if not set1 or not set2 :
			return 0.0
		return len(set1.intersection(set2)) / len(set1.union(set2))

	def _get_user_based_pred(self, user, movie_id, base_pred) :
		u_ratings = user.ratings
		if movie_id not in self.movie_user_ratings : return None
		other_users = self.movie_user_ratings[movie_id]
		if len(other_users) > self.max_comparisons :
			other_users = random.sample(other_users, self.max_comparisons)

		similarities = []
		for v_id, v_rating in other_users :
			v_ratings = self.users[v_id].ratings
			common_movies = u_ratings.keys() & v_ratings.keys()

			n_common = len(common_movies)
			if n_common == 0 : continue

			v_base_pred = self.global_mean + self.b_u.get(v_id, 0.0) + self.b_i.get(movie_id, 0.0)

			num = den_u = den_v = 0.0
			for m in common_movies :
				# Oceniamy odchylenie względem baseline'u danego usera dla konkretnego filmu
				u_m_base = self.global_mean + self.b_u.get(user.id, 0.0) + self.b_i.get(m, 0.0)
				v_m_base = self.global_mean + self.b_u.get(v_id, 0.0) + self.b_i.get(m, 0.0)

				diff_u = u_ratings[m] - u_m_base
				diff_v = v_ratings[m] - v_m_base
				num += diff_u * diff_v
				den_u += diff_u ** 2
				den_v += diff_v ** 2

			if den_u > 0 and den_v > 0 :
				sim = num / (math.sqrt(den_u) * math.sqrt(den_v))

				# Maksymalizujemy korelację dopiero od 10 wspólnych filmów
				significance_weight = min(n_common, 10) / 10.0
				sim *= significance_weight

				if sim > 0 :
					similarities.append((sim, v_rating, v_base_pred))

		if not similarities : return None
		similarities.sort(key=lambda x : x[0], reverse=True)
		top_k = similarities[:self.k_user]

		num = sum(sim * (v_rat - v_base) for sim, v_rat, v_base in top_k)
		den = sum(sim for sim, _, _ in top_k)
		return base_pred + (num / den) if den > 0 else None

	def _get_item_based_pred(self, user, movie_id) :
		if movie_id not in Movie.index : return None
		target_movie = Movie.index[movie_id]
		if not user.ratings : return None

		similarities = []
		for r_mid, r_rating in user.ratings.items() :
			if r_mid not in Movie.index : continue
			rated_movie = Movie.index[r_mid]

			genre_sim = self._jaccard_similarity(target_movie.genres, rated_movie.genres)
			title_sim = self._title_similarity(target_movie.name, rated_movie.name)

			if title_sim > 0.6 :
				combined_sim = 0.3 * genre_sim + 0.7 * title_sim
			else :
				combined_sim = genre_sim

			if combined_sim > 0 :
				similarities.append((combined_sim, r_rating))

		if not similarities : return None
		similarities.sort(key=lambda x : x[0], reverse=True)
		top_k = similarities[:self.k_item]

		num = sum(sim * rat for sim, rat in top_k)
		den = sum(sim for sim, rat in top_k)
		return num / den if den > 0 else None

	def rate(self, user, movie_id) :
		b_u_val = self.b_u.get(user.id, 0.0)

		b_i_val = self.b_i.get(movie_id, 0.0)
		baseline_pred = self.global_mean + b_u_val + b_i_val

		pred_u = self._get_user_based_pred(user, movie_id, baseline_pred)
		pred_i = self._get_item_based_pred(user, movie_id)

		if pred_u is not None and pred_i is not None :
			final_pred = 0.55 * pred_u + 0.30 * pred_i + 0.15 * baseline_pred
		elif pred_u is not None :
			final_pred = 0.8 * pred_u + 0.2 * baseline_pred
		elif pred_i is not None :
			final_pred = 0.7 * pred_i + 0.3 * baseline_pred
		else :
			final_pred = baseline_pred

		return max(0.5, min(5.0, final_pred))

	def __str__(self) :
		return 'System created by 155864 155916'