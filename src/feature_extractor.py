import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from loader import load_image
import antropy as ant

from scipy.spatial import distance


def get_combined_features(f1, f2, img1, img2):
    i1_flat = img1.flatten().astype(float)
    i2_flat = img2.flatten().astype(float)

    l1_dist = np.sum(np.abs(i1_flat - i2_flat))
    l2_dist = np.linalg.norm(i1_flat - i2_flat)

    cos_dist = distance.cosine(i1_flat + 1e-9, i2_flat + 1e-9)

    mu1, mu2 = np.mean(i1_flat), np.mean(i2_flat)
    sigma1, sigma2 = np.var(i1_flat), np.var(i2_flat)

    fid_dist = (mu1 - mu2) ** 2 + sigma1 + sigma2 - 2 * np.sqrt(sigma1 * sigma2)

    diff = np.abs(f1 - f2)
    dist_metrics = np.array([l1_dist, l2_dist, cos_dist, fid_dist])

    return np.concatenate([f1, f2, diff, dist_metrics])


class ImageFeatureExtractor:
    def __init__(self, glcm_levels=120, offset=59, lbp_radius=1, lbp_n_points=8):
        self.glcm_levels = glcm_levels
        self.offset = offset
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.MAX_DIST = np.sqrt(128 ** 2 + 128 ** 2)

    def extract_features(self, img):
        """Extracts all features and returns them as a dictionary."""
        # --- Pre-processing ---
        flat_img = img.flatten()

        # --- Shannon Entropy ---
        hist_ent, _ = np.histogram(flat_img, bins=64, density=True)
        hist_ent = hist_ent[hist_ent > 0]
        entropy = -(hist_ent * np.log2(hist_ent)).sum()

        # --- GLCM ---
        img_glcm = np.clip(img + self.offset, 0, self.glcm_levels - 1).astype(np.uint8)
        glcm = graycomatrix(img_glcm, [1], [0], levels=self.glcm_levels, symmetric=True, normed=True)

        # --- LBP ---
        img_lbp = np.clip(img + self.offset, 0, 120).astype(np.uint8)
        lbp_def = local_binary_pattern(img_lbp, self.lbp_n_points, self.lbp_radius, method='default')
        lbp_uni = local_binary_pattern(img_lbp, self.lbp_n_points, self.lbp_radius, method='uniform')

        # Calculate LBP Histogram Vector (Uniform)
        # Number of bins for uniform LBP is P + 2
        lbp_hist, _ = np.histogram(lbp_uni.ravel(), bins=np.arange(self.lbp_n_points + 3), density=True)

        # --- Spectral ---
        f_shift = np.fft.fftshift(np.fft.fft2(img))
        mag = np.abs(f_shift)
        pwr = mag ** 2
        rows, cols = img.shape
        y, x = np.ogrid[:rows, :cols]
        dists = np.sqrt((x - (cols // 2)) ** 2 + (y - (rows // 2)) ** 2)

        # Radial Slope
        d_flat = dists.ravel().astype(int)
        radial_prof = np.bincount(d_flat, weights=mag.ravel()) / (np.bincount(d_flat) + 1e-9)
        limit = min(128, len(radial_prof))
        # Ensure we don't take log of 0
        slope, _ = np.polyfit(np.log(np.arange(1, limit)), np.log(radial_prof[1:limit] + 1e-9), 1)

        # --- Compile Dictionary ---
        feats = {
            # Basic Statistics
            'mean': np.mean(img),
            'std': np.std(img),
            'min': np.min(img),
            'max': np.max(img),
            'kurtosis': kurtosis(flat_img),
            'skew': skew(flat_img),

            # Entropies
            'shannon_entropy': entropy,
            'permutation_entropy': ant.perm_entropy(flat_img, order=3, normalize=True),

            # GLCM
            'glcm_corr': graycoprops(glcm, 'correlation')[0, 0],
            'glcm_contrast': graycoprops(glcm, 'contrast')[0, 0],
            'glcm_homo': graycoprops(glcm, 'homogeneity')[0, 0],
            'glcm_energy': graycoprops(glcm, 'energy')[0, 0],
            'glcm_dissim': graycoprops(glcm, 'dissimilarity')[0, 0],

            # LBP Scalars
            'lbp_smooth': np.histogram(lbp_def.ravel(), bins=np.arange(2 ** self.lbp_n_points + 1), density=True)[0][0],
            'lbp_uniform_sum': np.sum(lbp_hist[:-1]),

            # Spectral
            'hf_energy': np.log(np.sum(pwr[dists > (0.8 * self.MAX_DIST)]) + 1e-9),
            'spectral_centroid': np.log((np.sum(dists * mag) / (np.sum(mag) + 1e-9)) + 1e-9),
            'lf_power': np.log(np.sum(pwr[dists < (0.1 * self.MAX_DIST)]) + 1e-9),
            'spectral_slope': slope
        }

        # Add the LBP Histogram Vector as individual features (for math compatibility)
        for i, val in enumerate(lbp_hist):
            feats[f'lbp_hist_{i}'] = val

        return feats

    def build_feature_dataframe(self, samples_df):
        samples_df = samples_df.copy()
        all_features = []
        for _, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Extracting Features"):
            img = load_image(row['id'])
            features = self.extract_features(img)
            features['id'] = row['id']
            all_features.append(features)

        feature_df = pd.DataFrame(all_features)
        cols = ['id'] + [c for c in feature_df.columns if c != 'id']
        return feature_df[cols].set_index('id')



    def build_dataset(self, features_df, pairs_df, type="train"):
        X_list = []
        y_list = []

        for _, row in tqdm(pairs_df.iterrows()):
            id1, id2 = row['id_noise_1'], row['id_noise_2']
            f1 = features_df.loc[id1]
            f2 = features_df.loc[id2]

            X_list.append(get_combined_features(f1, f2, load_image(id2), load_image(id1)))
            if 'label' in row:
                y_list.append(row['label'])

            if type != 'test':
                X_list.append(get_combined_features(f2, f1, load_image(id2), load_image(id1)))
                if 'label' in row:
                    y_list.append(row['label'])

        return np.array(X_list), np.array(y_list)