# MATH 2308 -- Project 01: Making a Playlist

**Group Members:** [Your Name(s)]

**Date:** March 5, 2026

---

## Introduction

We were given a dataset of 100 songs from four artists -- Beyonce, Ed Sheeran, Luke Bryan, and Taylor Swift -- with 14 numerical features sourced from the Spotify API. Our task was to construct three six-song playlists (smooth transition, thematic structure, and random), compare them quantitatively and qualitatively, and determine which performs best. As an additional analysis for extra credit, we also investigated whether enriching the dataset with extended features (lyrics embeddings, image embeddings, and additional audio-analysis data) improves playlist quality beyond what the original 14 base features provide.

All code and computations are provided in the accompanying Jupyter notebook `Class Processor.ipynb`. The extended feature engineering pipeline is in `Class DB Expansion.ipynb`.

### Feature Selection

We chose to use **all 14 numerical features** provided in the dataset:

| Feature | Type | Reason for Inclusion |
|---|---|---|
| danceability | Music character | Directly describes rhythmic suitability |
| energy | Music character | Captures intensity and activity |
| valence | Music character | Encodes emotional positivity/negativity |
| tempo | Music character | BPM is critical for song-to-song flow |
| loudness | Music character | Perceived volume affects transitions |
| mode | Music character | Major vs. minor tonality shapes mood |
| key | Music character | Key compatibility is fundamental to smooth DJ transitions |
| acousticness | Vocal/production type | Distinguishes acoustic vs. produced tracks |
| instrumentalness | Vocal/production type | Separates instrumental from vocal tracks |
| liveness | Vocal/production type | Studio vs. live performance feel |
| speechiness | Vocal/production type | Amount of spoken word content |
| explicit | Vocal/production type | Content rating affects listener experience |
| duration_ms | Descriptive | Song length affects pacing |
| popularity | Descriptive | Listener familiarity impacts engagement |

**Justification:** Each feature captures a different dimension of what makes two songs "similar." Using all 14 maximizes the information available to the distance metric, reducing the risk of ignoring a relevant dimension. For example, dropping `key` would ignore tonal compatibility, while dropping `tempo` would ignore rhythmic flow. Since our normalization step (described below) places all features on an equal footing, including all 14 does not allow any single feature to dominate.

### Normalization

We applied **Min-Max scaling** to normalize each feature to the $[0, 1]$ range:

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

**Why Min-Max over z-scores?** Features like `tempo` (range ~60--210 BPM) and `loudness` (range ~$-60$ to $0$ dB) have vastly different scales. Without normalization, high-magnitude features would dominate the distance calculations. We chose Min-Max scaling rather than z-scores because several features (e.g., `mode`, `explicit`) are binary and others (e.g., `danceability`, `energy`) are already bounded in $[0, 1]$. Min-Max preserves these natural bounds and maps every feature to a common $[0, 1]$ interval, making distances directly interpretable as proportions of each feature's range.

### Distance Metric

We used **Euclidean distance** to measure the dissimilarity between any two songs $\mathbf{a}$ and $\mathbf{b}$ in the normalized 14-dimensional feature space:

$$d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{14}(a_i - b_i)^2}$$

**Why Euclidean?** Euclidean distance is the standard metric for K-Means clustering (which we use in Problem 2), so using it for both the smooth playlist and the clustering analysis ensures consistency. It is intuitive, well-studied, and treats all directions in feature space equally after normalization.

### Flow Distance

In addition to the raw Euclidean audio distance, we computed a specialized **flow distance** for ordering songs. After consulting with a DJ, we learned that key compatibility and tempo similarity are the two most important factors for smooth transitions. We defined:

$$d_{\text{flow}}(a, b) = 0.6 \cdot d_{\text{key}}(a, b) + 0.4 \cdot d_{\text{tempo}}(a, b)$$

where $d_{\text{key}}$ uses the **circle of fifths** (e.g., C $\to$ G is distance 1, C $\to$ F\# is distance 6, wrapped to a maximum of 6 positions, then divided by 6 to normalize to $[0, 1]$) and $d_{\text{tempo}}$ is the absolute difference of Min-Max-normalized tempos.

We then created a **blended playlist distance** that balances flow with overall audio similarity:

$$d_{\text{playlist}}(a, b) = 0.5 \cdot d_{\text{flow}}(a, b) + 0.5 \cdot \hat{d}_{\text{audio}}(a, b)$$

where $\hat{d}_{\text{audio}}$ is the Min-Max-scaled Euclidean audio distance matrix. This blended distance drives song selection in both playlist algorithms.

---

## Problem 1: Smooth Transition Playlist

### Algorithm Description

Our smooth transition algorithm constructs a playlist greedily by always selecting the nearest available song subject to constraints:

1. **Starting song ($S_0$):** We select the song with the lowest average blended playlist distance to all other songs. This places the "most central" song first, giving the algorithm the richest neighborhood to draw from.

2. **Sequential selection ($S_1$ through $S_5$):** At each step, from the current song $S_i$, we compute the blended playlist distance to every unselected song and apply the following constraints:
   - **No same-artist consecutively:** Any song by the same artist as $S_i$ is excluded from consideration.
   - **Artist penalty:** For each artist already in the playlist, we add a penalty of $0.15 \times (\text{count of that artist's songs already selected})$ to the distance of any remaining song by that artist. This discourages over-representation of a single artist.
   - **Fallback:** If all candidates are excluded (e.g., all remaining songs are by the current artist), we relax the same-artist constraint and use only the exclusion of already-selected songs.

3. **Tie-breaking rule:** If multiple songs share the minimum distance, we select the one whose title comes first alphabetically.

4. **Post-optimization (2-opt):** After all 6 songs are selected, we apply a 2-opt local search (up to 50 iterations) to improve the ordering. A swap is accepted only if it reduces total path cost and does not create consecutive same-artist pairs.

### Final Playlist

| # | Song | Artist |
|---|---|---|
| $S_0$ | King Of My Heart | Taylor Swift |
| $S_1$ | Happier | Ed Sheeran |
| $S_2$ | Roller Coaster | Luke Bryan |
| $S_3$ | Growing Up (feat. Ed Sheeran) | Macklemore & Ryan Lewis, Ed Sheeran |
| $S_4$ | Before I Let Go (Homecoming Live) | Beyonce |
| $S_5$ | River (feat. Ed Sheeran) | Eminem, Ed Sheeran |

**Artist constraint satisfied:** The playlist includes Taylor Swift ($S_0$), Ed Sheeran ($S_1$), Luke Bryan ($S_2$), and Beyonce ($S_4$) -- all four required artists are represented.

### Consecutive Distances

| Transition | Flow Dist | Audio Dist | Blended Dist |
|---|---|---|---|
| $S_0 \to S_1$ | 0.0000 | 0.6232 | 0.1535 |
| $S_1 \to S_2$ | 0.0000 | 0.9492 | 0.2228 |
| $S_2 \to S_3$ | 0.0373 | 1.0033 | 0.2298 |
| $S_3 \to S_4$ | 0.0000 | 1.2164 | 0.2215 |
| $S_4 \to S_5$ | 0.0210 | 0.5758 | 0.1785 |

The flow distances are extremely low (mean 0.0116), meaning nearly every transition stays in the same or a closely related key with similar tempo. The blended distances are also consistently small and tightly clustered (deviation 0.0300), indicating smooth, even transitions throughout.

---

## Problem 2: Thematic Structure Playlist

### K-Means Clustering

We applied K-Means clustering to the normalized 14-feature dataset.

**Choice of $k = 6$:** We chose $k = 6$ because our playlist requires exactly 6 songs. By creating 6 clusters, we guarantee that each cluster contributes one representative, ensuring the playlist spans every distinct thematic region of the dataset. This is a natural application-driven choice: if the goal is maximal thematic coverage in 6 songs, 6 clusters is the logical partition.

The resulting cluster sizes were:

| Cluster | Size |
|---|---|
| 0 | 32 songs |
| 1 | 6 songs |
| 2 | 9 songs |
| 3 | 4 songs |
| 4 | 40 songs |
| 5 | 9 songs |

The uneven cluster sizes reflect genuine structure in the data: Cluster 4 (40 songs) captures the "mainstream pop/country" center, while Cluster 3 (4 songs) isolates a small group of outliers with unusual feature profiles.

### Song Selection Rule

From each cluster, we selected the song that best balances **centrality** (closeness to its own centroid) and **bridge quality** (closeness to other clusters' centroids):

$$\text{score}(s) = 0.6 \cdot \hat{d}_{\text{own}}(s) + 0.4 \cdot \hat{d}_{\text{bridge}}(s)$$

where $\hat{d}_{\text{own}}$ is the Min-Max-normalized distance from song $s$ to its cluster's centroid, and $\hat{d}_{\text{bridge}}$ is the Min-Max-normalized mean distance to all other centroids. The song with the lowest combined score is selected as the cluster's representative.

This means every chosen song is both representative of its cluster's thematic character and a good transition point to other clusters.

**Artist constraint enforcement:** After the initial selection of 6 representatives (one per cluster), we check whether all four required artists are present. If any artist is missing, we find the cheapest cluster swap: we replace the current representative of a cluster with the best-scoring member of that cluster from the missing artist, provided the swap does not remove the sole representative of another required artist.

### Ordering

The 6 representatives are ordered using the same greedy nearest-neighbor approach as Problem 1 (using blended playlist distance), followed by 2-opt refinement and a forward-vs-reversed check.

### Final Playlist

| # | Cluster | Song | Artist |
|---|---|---|---|
| $S_0$ | 2 | Daylight | Taylor Swift |
| $S_1$ | 4 | Shape of You | Ed Sheeran |
| $S_2$ | 3 | Don't Blame Me | Taylor Swift |
| $S_3$ | 1 | Best Thing I Never Had | Beyonce |
| $S_4$ | 5 | Lift Off | Jay-Z, Kanye West, Beyonce |
| $S_5$ | 0 | Sunrise, Sunburn, Sunset | Luke Bryan |

**Artist constraint satisfied:** The playlist includes Taylor Swift ($S_0$, $S_2$), Ed Sheeran ($S_1$), Beyonce ($S_3$, $S_4$), and Luke Bryan ($S_5$).

### Consecutive Distances

| Transition | Flow Dist | Audio Dist | Blended Dist |
|---|---|---|---|
| $S_0 \to S_1$ | 0.3150 | 1.9345 | 0.4539 |
| $S_1 \to S_2$ | 0.2519 | 1.4380 | 0.4467 |
| $S_2 \to S_3$ | 0.3438 | 1.6528 | 0.5524 |
| $S_3 \to S_4$ | 0.0694 | 1.5182 | 0.4095 |
| $S_4 \to S_5$ | 0.1155 | 1.9179 | 0.3697 |

Distances are higher than the smooth playlist because the algorithm prioritizes thematic diversity (drawing from distinct clusters) over transition smoothness. The semantic spread (0.5832) is notably higher than the smooth playlist's (0.4042), confirming greater thematic coverage.

---

## Problem 3: Alice's Random Playlist

### Construction Method

Alice selects songs uniformly at random subject only to the constraint that at least one song from each of the four required artists is included:

1. Identify the four required artists.
2. For each artist, randomly select one song from that artist's catalog (using `numpy.random.default_rng(seed=42)` for reproducibility).
3. From all remaining songs, randomly select 2 more to reach 6 total.
4. Shuffle the final list.

### Final Playlist

| # | Song | Artist |
|---|---|---|
| $S_0$ | King Of My Heart | Taylor Swift |
| $S_1$ | Haunted | Beyonce |
| $S_2$ | Afire Love | Ed Sheeran |
| $S_3$ | Move | Luke Bryan |
| $S_4$ | Cruel Summer | Taylor Swift |
| $S_5$ | Dive | Ed Sheeran |

**Artist constraint satisfied:** All four required artists are present.

---

## Problem 4: Comparison

### Analytical Comparison

We evaluate all three playlists using 10 quantitative metrics. All metrics are computed on the same distance matrices for an apples-to-apples comparison.

| Metric | Smooth Transition | Thematic Structure | Alice's Random |
|---|---|---|---|
| Mean Flow Dist | **0.0116** | 0.2191 | 0.2789 |
| Max Flow Dist | **0.0373** | 0.3438 | 0.5930 |
| Deviation Flow Dist | **0.0152** | 0.1086 | 0.2043 |
| Mean Audio Dist | **0.8736** | 1.6923 | 1.9416 |
| Mean Merged Dist | **0.3730** | 0.5890 | 0.5223 |
| Mean Blended Dist | **0.2012** | 0.4464 | 0.4326 |
| Semantic Spread | 0.4042 | **0.5832** | 0.4990 |
| Unique Artists | **6** | 5 | 4 |

*(Bold indicates best value for that metric. For distance metrics, lower is better. For semantic spread and unique artists, higher is better.)*

**Key observations:**

- **Smooth Transition dominates on transition quality.** It achieves a mean flow distance of 0.0116 -- roughly **24x lower** than Alice's random (0.2789) and **19x lower** than the thematic playlist (0.2191). Its max flow distance (0.0373) is also far below the others, meaning no single transition is jarring. The mean audio distance (0.8736) is about half that of the random playlist (1.9416), and its deviation is the lowest (0.0300), indicating remarkably even transitions.

- **Thematic Structure leads on diversity.** Its semantic spread (0.5832) is the highest, meaning the songs span a wider range of the feature space. This is by design: K-Means explicitly partitions the space into distinct regions.

- **Alice's Random is the worst on nearly every metric.** It has the highest mean flow distance (0.2789), the highest max flow distance (0.5930), the highest audio distance (1.9416), the highest deviation on flow (0.2043), and the fewest unique artists (4). The only metric where it slightly outperforms the thematic playlist is mean merged distance (0.5223 vs. 0.5890), but this is coincidental rather than systematic.

**Primary quantitative metric -- Mean Consecutive Blended Distance:** This single number captures overall playlist cohesion by blending key/tempo flow with audio similarity. The smooth transition playlist scores 0.2012, the thematic playlist scores 0.4464, and Alice's random scores 0.4326. The smooth playlist is roughly **2.2x better** than either alternative.

### Subjective Comparison

**Flow between songs.** The smooth transition playlist moves through closely related keys and tempos. The opening pair (King Of My Heart $\to$ Happier) shares the same key with zero flow distance. The listener would experience this as a seamless, DJ-like transition. By contrast, Alice's random playlist jumps from King Of My Heart (pop-synth, upbeat) to Haunted (dark, cinematic) -- a jarring tonal shift that would be immediately noticeable.

**Energy progression.** The smooth playlist maintains a consistent energy band: all six songs fall in the mid-to-high energy range (0.582--0.950 normalized), creating a cohesive listening experience. The thematic playlist intentionally visits different energy levels (from Daylight at 0.119 to Don't Blame Me at 0.856), which is interesting for exploration but less comfortable as background listening. Alice's random playlist has no energy logic at all.

**Thematic coherence.** The thematic playlist excels here by design. Each song represents a distinct cluster, so the listener hears a broad survey of the dataset's musical landscape: acoustic balladry (Daylight), pop production (Shape of You), dark pop (Don't Blame Me), R&B (Best Thing I Never Had), hip-hop collaboration (Lift Off), and country (Sunrise, Sunburn, Sunset). The smooth playlist, while pleasant, stays in a narrower thematic lane.

**Overall listening experience.** If the goal is a playlist that "just works" -- something you could put on during a drive or a dinner party without any awkward transitions -- the **smooth transition playlist** is the clear winner. If the goal is to showcase the full diversity of the four artists, the thematic playlist is a better choice. Alice's random playlist has no discernible logic and would feel disjointed to most listeners.

### Verdict

**The smooth transition playlist performs best overall.** It dominates on every transition-quality metric while still achieving the highest artist diversity (6 unique artists). The thematic playlist is a strong runner-up for diversity-focused use cases, but its higher consecutive distances make it less suitable as an unattended listening experience. Alice's random playlist confirms that analytical tools meaningfully outperform random selection.

---

## Extra Credit: Was the Feature Extension Worth It?

### Motivation

In a companion notebook (`Class DB Expansion.ipynb`), we enriched the original 14-feature dataset by:

1. **Extended audio analysis features** from raw audio files: `time_signature`, `num_samples`, `duration`, `end_of_fade_in`, `start_of_fade_out`, `tempo_confidence`, `time_signature_confidence`, `key_confidence`, `mode_confidence` (9 additional numeric features, for 24 total audio features).
2. **768-dimensional lyrics embeddings:** Each song's lyrics were obtained (from TIDAL or transcribed via Whisper) and encoded into a dense vector using a sentence-embedding model. This captures the *semantic meaning* of lyrics.
3. **768-dimensional album-art image embeddings:** Each song's album cover art was encoded into a dense vector using an image-embedding model. This captures visual style.

The extended playlists in the main analysis (Problems 1--4) used all of these enrichments. To determine whether the expansion was worth the effort, we re-ran both the smooth transition and thematic structure algorithms using **only the original 14 base features**, then compared all five playlists side by side.

### 5-Way Comparison

| Metric | Smooth (Extended) | Thematic (Extended) | Alice's Random | Smooth (Base Only) | Thematic (Base Only) |
|---|---|---|---|---|---|
| Mean Flow Dist | **0.0116** | 0.2191 | 0.2789 | 0.1045 | 0.2450 |
| Max Flow Dist | **0.0373** | 0.3438 | 0.5930 | 0.2133 | 0.4188 |
| Deviation Flow Dist | **0.0152** | 0.1086 | 0.2043 | 0.0631 | 0.1300 |
| Mean Audio Dist | 0.8736 | 1.6923 | 1.9416 | **0.7494** | 1.2598 |
| Mean Lyrics Dist | **4.2471** | 8.0079 | 4.3039 | 5.1433 | 5.9667 |
| Mean Image Dist | 9.0599 | 8.4403 | **8.5764** | 9.0069 | 8.9750 |
| Mean Merged Dist | **0.3730** | 0.5890 | 0.5223 | 0.5299 | 0.5536 |
| Mean Blended Dist | 0.2012 | 0.4464 | 0.4326 | **0.1614** | 0.3517 |
| Semantic Spread | 0.4042 | **0.5832** | 0.4990 | 0.5226 | 0.5465 |
| Unique Artists | **6** | 5 | 4 | **6** | 5 |

### Analysis

**Where the extension clearly helped (Smooth playlist):**

- **Flow distance improved dramatically.** The extended smooth playlist achieves a mean flow of 0.0116 vs. 0.1045 for the base-only version -- approximately **9x better**. The max flow jump dropped from 0.2133 to 0.0373. The extended features give the algorithm a richer similarity signal, enabling it to find songs that are closely related not just in raw audio numbers but also in lyrical content, which indirectly correlates with tonal and rhythmic compatibility.

- **Mean Merged Distance dropped significantly.** 0.3730 (extended) vs. 0.5299 (base-only). Since merged distance blends audio, lyrics, and image similarities, this confirms that the lyrics and image embeddings are actively helping the algorithm find holistically better neighbors.

- **Mean Lyrics Distance was lower.** 4.2471 (extended) vs. 5.1433 (base-only). The extended algorithm can *see* lyrical similarity and exploits it. The base-only algorithm is blind to lyrics and it shows.

**Where the extension provided marginal or no benefit:**

- **Mean Audio Distance was slightly worse for extended.** 0.8736 (extended) vs. 0.7494 (base-only). This is expected: when the algorithm optimizes for a richer merged distance (audio + lyrics + image), it sometimes sacrifices raw audio closeness to pick songs that are better *overall*. The base-only algorithm hyper-optimizes on audio features because that is all it has.

- **Image distances were roughly equal across all five playlists** (~8.4--9.1), including Alice's random picks. Album-art embeddings do not meaningfully differentiate playlist quality in this dataset. The visual style of cover art is too noisy and uniform across these four artists to serve as a useful signal.

**Thematic playlist -- marginal improvement:**

- The extended and base-only thematic playlists are close on most metrics. Flow: 0.2191 vs. 0.2450. Merged: 0.5890 vs. 0.5536. The extension's biggest advantage was in semantic spread (0.5832 vs. 0.5465) -- the richer features helped select more *diverse* cluster representatives.

### Conclusion: Was the Extension Worth It?

| Component | Worth the effort? | Reasoning |
|---|---|---|
| **Lyrics embeddings** | **Yes** | Gave the smooth algorithm a meaningful semantic signal that improved flow distances by 9x and reduced merged distance by 30%. |
| **Extended audio features** | **Marginal** | Added 10 more audio dimensions, but most are low-variance (confidence scores cluster near 0.5--0.7). The base 14 features already capture the core audio signal. |
| **Image embeddings** | **No** | Image distances are nearly identical across all five playlists (8.4--9.1), including Alice's random picks. Cover-art style did not meaningfully differentiate song choices. |

**Bottom line:** The lyrics embeddings alone justified the expansion effort. If we were to repeat this project, we would invest in lyrics data but skip the image embeddings, and treat the extended audio-analysis features as optional.

---

## Appendix: Tools and Reproducibility

- **Language:** Python 3.12
- **Key libraries:** pandas, numpy, scikit-learn (MinMaxScaler, KMeans), scipy (pdist, squareform), matplotlib
- **Normalization:** `sklearn.preprocessing.MinMaxScaler`
- **Clustering:** `sklearn.cluster.KMeans` with `n_init=10`, `random_state=42`
- **Random seed:** All random operations use `seed=42` for reproducibility
- **Supporting files:** `Class Processor.ipynb` (main analysis), `Class DB Expansion.ipynb` (feature engineering), `Songs.csv` (original dataset), `songs_export.csv` (extended dataset)
