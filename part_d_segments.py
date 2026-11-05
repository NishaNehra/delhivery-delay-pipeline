"""Part D - hub profiles, K-Means segments and the tier-versus-segment check"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
K_VALUES = [2, 3, 4, 5]
N_SEGMENTS = 3
SEGMENT_NAMES = ["Standard", "Watch", "Escalate"]  # lowest -> highest mean delay rate
TIER_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}


def fit_kmeans(matrix, k):
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix)


def across_over_within(matrix, labels):
    i, j = np.triu_indices(len(matrix), k=1)
    dist = np.linalg.norm(matrix[i] - matrix[j], axis=1)
    same = labels[i] == labels[j]
    return dist[~same].mean() / dist[same].mean()


def main():
    os.makedirs("outputs", exist_ok=True)
    table = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # 1. hub profile in ascending hub_id
    profile = (table.groupby("hub_id")
               .agg(shipments=("shipment_id", "size"), mean_delivery_hours=("delivery_hours", "mean"),
                    delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
                    mean_n_stops=("n_stops", "mean"))
               .sort_index()
               .reset_index())
    profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

    # 2. K-Means with K = 3 on the scaled profile
    scaled = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])
    labels = fit_kmeans(scaled, N_SEGMENTS)

    # 3. separation evidence for K
    sep_rows = []
    for k in K_VALUES:
        k_labels = labels if k == N_SEGMENTS else fit_kmeans(scaled, k)
        sizes = sorted(np.bincount(k_labels), reverse=True)
        sep_rows.append({"K": k, "across_over_within": round(float(across_over_within(scaled, k_labels)), 4),
                         "cluster_sizes": "-".join(str(s) for s in sizes)})
    pd.DataFrame(sep_rows).to_csv("outputs/part_d_separation.csv", index=False)

    # 4. name clusters by mean delay rate then comparison with tier label
    profile["cluster"] = labels
    cluster_order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
    profile["segment"] = profile["cluster"].map(dict(zip(cluster_order, SEGMENT_NAMES)))

    segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]],
                                                    on="hub_id", how="left", validate="one_to_one")
    segments.to_csv("outputs/hub_segments.csv", index=False)

    seg_profile = (profile.groupby("segment")
                   .agg(hubs=("hub_id", "size"), **{c: (c, "mean") for c in PROFILE_COLUMNS})
                   .reindex(SEGMENT_NAMES)
                   .reset_index())
    seg_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

    mismatch = segments[segments["segment"] != segments["hub_tier"].map(TIER_SEGMENT)]
    mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

    print(pd.DataFrame(sep_rows).to_string(index=False))
    print(seg_profile.round(4).to_string(index=False))
    print(segments.to_string(index=False))
    print(mismatch.to_string(index=False))


if __name__ == "__main__":
    main()
