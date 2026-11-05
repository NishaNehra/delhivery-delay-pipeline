# delhivery-delay-pipeline

This project builds a delay pipeline for Delhivery's regional operations team. It cleans historical parcel data, predicts how long a parcel will take, flags parcels that are likely to miss their promised delivery time, groups the 24 hubs into operating segments, and turns a batch of new parcels into an action list for ops.

The data is synthetic. It is created by `generate_data.py`, not taken from real Delhivery shipments.

The project answers three questions:

1. How many hours will a parcel take? (Part B, regression)
2. Which parcels will miss their promised time? (Part C, classification)
3. Which hubs behave alike, and does their tier label match? (Part D, clustering)

## How to run

Run these commands one by one from the main folder of a fresh clone (Python 3 with `venv`):

```bash
git clone <repo-url> delhivery-delay-pipeline
cd delhivery-delay-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate_data.py          # prints: 24 1224 48; rewrites data/hubs.csv, raw_shipments.csv, new_shipments.csv
python part_a_clean_eda.py       # writes data/modelling_table.csv and outputs/part_a_*.csv
python part_b_regression.py      # writes outputs/part_b_*.csv
python part_c_classification.py  # writes outputs/part_c_*.csv
python part_d_segments.py        # writes outputs/hub_profile.csv, part_d_separation.csv, hub_segments.csv,
                                 #        segment_profile.csv, tier_vs_segment_mismatch.csv
python package_model.py          # writes models/delay_model.pkl, delay_threshold.csv, feature_order.csv,
                                 #        impute_values.csv and outputs/part_d_reload_check.csv
python predict_delay.py          # writes outputs/new_shipment_actions.csv
```

Some notes:

- `models/delay_model.pkl` is created again by running `python package_model.py`.
- The saved pickle only loads reliably with the scikit-learn version pinned in `requirements.txt`. If your installed versions are different, run `package_model.py` again instead of loading my committed pickle (the grader does the same).
- I ran `predict_delay.py` twice and got exactly the same `outputs/new_shipment_actions.csv` both times.

## Part A — Data findings

First I cleaned the messy export: removed duplicate rows, fixed the different spellings of `service_type`, turned the declared value text into a number, treated weights above 100 kg as scanner errors, and filled gaps with the median. The numbers below come from `outputs/part_a_checks.csv`, `outputs/part_a_delay_by_group.csv` and `outputs/part_a_correlations.csv`.

1. **Express is late most often.** 119 of 243 Express parcels were delayed, a delay rate of **0.4897**. Standard was 0.2056 and Economy 0.0857. Overall, 278 of 1,200 parcels were late (0.2317). I expected Express to be the most reliable, but it has the tightest promise (36 hours), so there is less spare time if anything goes wrong.
2. **Higher tier, more delays.** Tier 1 hubs have a delay rate of **0.0400**, Tier 2 **0.2150** and Tier 3 **0.4400**. The jump from Tier 1 to Tier 3 is very large.
3. **Longer distance, more delays.** The delay rate goes up in every distance band: **0.0219** (0-200 km), **0.1368** (201-400 km), **0.4364** (401-600 km) and **0.8523** (601+ km). Most parcels going more than 600 km were late.
4. **Columns most related to `delivery_hours`.** I sorted the correlations by their absolute value (ignoring the sign). The top three are `promised_hours` (**0.8372**), `distance_km` (**0.4013**) and `n_stops` (**0.3247**). All three are positive, so higher values mean longer delivery. `promised_hours` is first simply because Economy parcels take longer than Express ones.
5. **Why I used the log of declared value.** After filling the gaps, `declared_value_inr` has a mean of INR 1,533.05 but a median of only INR 1,121.50 (`part_a_checks.csv`: 1533.0542 and 1121.5). The mean is much bigger because a few expensive parcels pull it up (the column is skewed). After taking the log, the mean is 7.0277 and the median 7.0224, almost the same. My course taught that a log helps with money columns like this, so a few very expensive parcels do not dominate the feature.

## Part B — Model results (regression)

The goal here is to predict `delivery_hours`. I split the 1,200 parcels into 840 for training, 180 for validation (to compare models) and 180 for test (used only once at the end). I fitted the scaler on the training rows only, so the model never "sees" the test rows. `part_b_checks.csv` proves this: after scaling, the largest column mean is 0.0 on the training rows but 0.2187 on the test rows.

Two terms I use below: **RMSE** is roughly the typical error in hours, and **R²** shows how much of the variation the model explains (1 is perfect, 0 is no better than guessing the average). The numbers come from `outputs/part_b_metrics.csv`, `part_b_tree_depth.csv`, `part_b_gb_learning_rate.csv` and `part_b_cv.csv`.

1. **Baseline.** If I just guess the training average for every parcel, the validation R² is **−0.0212** and the RMSE is 18.8552 hours. An R² around zero means this guess explains nothing. It is only a starting point that a real model has to beat.
2. **Best model on validation.** Linear regression had the lowest validation RMSE, **4.8912 hours** (R² 0.9313). Gradient boosting was very close (4.9165 hours, R² 0.9306), then random forest (5.4784 hours, R² 0.9138) and the depth-6 decision tree (6.2796 hours, R² 0.8867). I then tested linear regression once on the test rows: RMSE 4.5235 hours, R² **0.9493**. In simple words, the estimate is usually off by about 4 to 5 hours.
3. **Tree depth: underfitting and overfitting.** Very shallow trees (depth 1 to 3) underfit (high bias): the error is high on both training and validation, and the two are close (depth 1: train 11.6334, validation 12.1324, gap 0.50). Depth 6 had the lowest validation error (**6.2796**). From depth 8 the trees overfit (high variance): the training error keeps falling but the validation error goes up. With no depth limit the training error is 0.0000 but validation is 7.7890 (gap 7.79), so the tree has basically memorised the training rows.
4. **Learning rate for gradient boosting.** 0.1 worked best (validation R² **0.9306**). With 0.01 the model learns too slowly, and 100 trees are not enough to reach a good fit (R² 0.7127). With 1.0 every tree takes too big a step and overshoots, so the model starts fitting noise (R² 0.8440).
5. **Cross-validation gave a different winner.** In 5-fold cross-validation on the 1,020 training plus validation rows, gradient boosting scored higher (mean R² **0.9352**, std 0.0044) than linear regression (mean R² **0.9261**, std 0.0048). On the single validation split, linear regression was ahead by only 0.0253 hours. Cross-validation checks 5 different splits instead of one split of 180 rows, so I trust it more. If I had to ship one model, I would choose **gradient boosting**. The test score above is for linear regression only, because the brief says to test the validation winner exactly once.

## Part C — Model results (classification)

Now the question is yes or no: will the parcel be late (`is_delayed`)? I used the same kind of split as Part B, but stratified, so each part keeps the same share of late parcels: 840 / 180 / 180 parcels, with 195 / 42 / 41 late.

Terms I use below: **precision** is how many of the parcels we flag are really late, **recall** is how many of the late parcels we manage to flag, and **F1** combines the two. For this task recall matters most, because a missed late parcel costs more than an extra phone call. The numbers come from `outputs/part_c_models.csv`, `part_c_feature_scores.csv`, `part_c_selectk.csv`, `part_c_tuning.csv`, `part_c_thresholds.csv`, `part_c_final_test.csv` and `part_c_final_config.csv`.

1. **Why accuracy alone is misleading.** A "model" that says every parcel is on time gets an accuracy of **0.7667** on validation (TN 138, FN 42), which looks good. But its recall is 0: it misses all 42 late parcels, so ops would never call anyone. This was the clearest lesson for me from this part. All my trained models had better accuracy (0.8833 to 0.9111), but I compared them mainly on F1: logistic regression 0.8046, random forest 0.8000 and the depth-6 decision tree 0.7470.
2. **The bookkeeping columns are noise.** The five columns like `label_print_batch` and `tape_roll_no` are just random numbers, but the random forest still gave them **0.2252** of its total importance. The F-test (`f_classif`) handled them better: none of them is in the top 7. The best of them is `label_print_batch` at rank 8 with an F-score of only 4.5847. The top 7 are `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag` and `sale_event_flag`, all with F-scores above 36.
3. **How many columns to keep.** Validation F1 was highest with 7 columns (**0.8333**). With 5 columns it dropped to 0.6988, because `rain_flag` and `sale_event_flag` are removed and they are useful. With more than 7 columns the weak and noise columns come back, and F1 goes down: 0.8140 (10 columns), 0.8049 (12) and 0.8000 (17).
4. **Tuning did not help.** Grid search (best CV accuracy 0.8952, `max_depth` 6, 100 trees) and randomized search (0.8905, `max_depth` 6, 50 trees) both scored *lower* on validation F1 (0.7750 and 0.7901) than the forest with default settings on the same 7 columns (**0.8333**). So I kept the default forest (`default_forest_same_7_features`). I think this is because the searches pick the best accuracy, not the best F1, and the validation set has only 42 late parcels, so these differences come down to two or three parcels. It does not mean tuning is bad; it means 180 validation rows are too few to see a clear difference.
5. **Choosing the alert threshold.** The model gives a probability, and a parcel is flagged when the probability is at least the threshold. The rule was: pick the highest threshold where validation recall is still at least 0.80. That is **0.5** (precision 0.7955, recall 0.8333, F1 0.8140). At 0.6 precision would go up to 0.8889, but recall would fall to 0.7619 (10 late parcels missed instead of 7), which breaks the rule. Small note: at 0.5 the F1 is 0.8140 and not the 0.8333 from tuning, because two validation parcels have a probability of exactly 0.5. My rule ("at least 0.5") flags them, but sklearn's `predict()` does not.
6. **Final test result.** On the 180 test parcels with threshold 0.5: precision **0.6889**, recall **0.7561**, F1 **0.7209** (TP 31, FN 10, FP 14). Honestly, test recall is lower than validation recall (0.8333) and below the 0.80 target. The test set has only 41 late parcels, so a few parcels make a big difference here. I chose the threshold on validation and did not change anything after seeing the test result.

## Operations recommendation

This section is for the regional operations head. In Part D I grouped the 24 hubs into 3 segments with K-Means, using each hub's average delivery hours, delay rate, distance and number of stops. I named the segments by their delay rate: Standard (lowest), Watch (middle) and Escalate (highest). The numbers come from `outputs/hub_profile.csv`, `segment_profile.csv`, `tier_vs_segment_mismatch.csv`, `part_d_separation.csv`, `part_c_final_test.csv` and `new_shipment_actions.csv`.

1. **Escalate hubs (9 hubs, average delay rate 0.4600).** HUB13 Nagpur 0.46, HUB15 Kochi 0.36, HUB17 Patna 0.46, HUB18 Ranchi 0.38, HUB19 Guwahati 0.62, HUB20 Raipur 0.48, HUB21 Bhubaneswar 0.40, HUB23 Siliguri 0.44 and HUB24 Agartala 0.54. These hubs have long routes (470.4757 km on average) and many stops (84.3444). Flagged parcels from these hubs should go straight to the hub manager. For comparison, Watch hubs average 0.1550 and Standard hubs 0.0257.
2. **Four hubs do not behave like their tier.**
   - **HUB07 Kolkata** is Tier 1 but falls in Watch (delay rate 0.14, average distance 304.736 km). It works more like a regional hub than a metro hub.
   - **HUB13 Nagpur** is Tier 2 but falls in Escalate (0.46, 473.32 km). It is as stretched as the Tier 3 hubs.
   - **HUB15 Kochi** is Tier 2 but falls in Escalate (0.36, 466.514 km). Also stretched.
   - **HUB22 Dehradun** is Tier 3 but falls in Watch (0.20, 316.738 km). It does better than its label says.

   My suggestion: choose each hub's playbook from its segment, not from its tier label, and ask the network team to review these four labels. The classifier also uses `hub_tier` as an input, so a wrong label can make its predictions worse for these hubs.
3. **What the 0.5 alert threshold means in practice.** On the test set (`part_c_final_test.csv`), about **7 out of every 10 flagged parcels are really late** (precision 0.6889), and about **7.5 out of every 10 late parcels get flagged** (recall 0.7561). So about 3 in 10 calls go to parcels that would have arrived on time. I think that is acceptable, because missing a late parcel costs more than an extra call.
4. **Today's 48 new parcels.** **8 are flagged.** **6 of them need a hub-manager escalation**: SHP90002, SHP90003, SHP90011, SHP90025, SHP90041 and SHP90046 (all from Escalate hubs). **2 need a proactive customer call**: SHP90005 (HUB22) and SHP90007 (HUB09), both Watch hubs. The other 40 need no action.
5. **Why I kept K = 3.** In `part_d_separation.csv` the across-to-within distance ratio (how far apart the groups are compared with how tight each group is) goes up from 2.894 (K = 2) to 5.3928 (K = 3), 6.3047 (K = 4) and 6.8866 (K = 5). So more clusters always look "better" by this number, partly because smaller groups are naturally tighter. But K = 4 gives group sizes 8-7-5-4 and K = 5 gives 8-7-4-3-2, so both have a group of fewer than 6 hubs, which is too small for its own playbook. K = 3 gives 9-8-7, has the biggest jump in the ratio, and matches the three playbooks operations actually uses.

## Production plan

I have not put a model into production before, so this plan is based on what I learned in my course and how I would apply it to this project.

**(a) How the repository covers the production-readiness points**

| Characteristic | Where it is in my repository |
|---|---|
| Accuracy | Every model is compared with a simple baseline, and the final models are tested once on held-out rows: `outputs/part_c_final_test.csv` (F1 0.7209) and `outputs/part_b_metrics.csv` (test R² 0.9493). |
| Reliability | `package_model.py` saves the model, loads it back and checks that both give the same probabilities (`outputs/part_d_reload_check.csv`, largest difference 0.0). Missing values in new parcels are filled with the stored medians, so the script does not break on gaps. |
| Reproducibility | `random_state=42` is used everywhere, versions are pinned in `requirements.txt`, and running `predict_delay.py` twice gives exactly the same file. |
| Consistency | Training and prediction use the same two functions, `clean_shipments` and `build_features` from `src/shipment_prep.py`, plus the saved medians (`models/impute_values.csv`) and column order (`models/feature_order.csv`). |
| Maintainability | One script per Part, and the shared constants (`PROMISED_HOURS`, `FEATURES`, `NOISE_COLUMNS`) are in one file. The final settings and threshold are saved in CSV files, not typed into the code. |
| Performance (speed) | Running `python predict_delay.py` for the 48 new parcels takes about 1 second, including loading the model. |
| Monitorability | `outputs/new_shipment_actions.csv` keeps the probability, flag and action for every parcel, so later we can compare them with what really happened. |

**(b) What I would monitor**

| What to check | Starting value | When to take action |
|---|---|---|
| Weekly share of delivered parcels that were late | 0.2317 (Part A) | outside 0.18–0.29 for two normal weeks in a row |
| Weekly share of parcels flagged at threshold 0.5 | about a quarter (validation at 0.5: TP 35 + FP 9 out of 180) | below 0.15 or above 0.35 |
| Recall on parcels whose real outcome is now known (last 4 weeks) | 0.7561 (test) | below 0.70 |
| Precision on the same parcels | 0.6889 (test) | below 0.55 (too many unnecessary calls) |
| Each hub's delay rate compared with its segment | segment averages 0.0257 / 0.1550 / 0.4600 | a hub is closer to another segment's average for two months → run `part_d_segments.py` again |

**(c) Retraining, the sale window and data drift**

- **Retraining schedule.** Once a month, retrain on the most recent delivered parcels by running Parts A to D and `package_model.py` again. Only replace the old model if the new one has at least the same test F1 and recall. If any of the checks above goes outside its range, retrain straight away.
- **The 15–28 September sale (`sale_event_flag = 1`).** During the sale, delays and flags are expected to go up. So these weeks should be compared with the previous sale, not with normal weeks, otherwise the alerts would fire for no real reason. The training data should also always include at least one past sale period, so the model learns how the sale affects delays.
- **What data drift would look like here.** The inputs start to change: for example a hub's average `distance_km` or `n_stops` moves away from `hub_profile.csv`, the share of COD parcels changes, or new hubs are added. Or the delay rate per service type moves away from the Part A table. A warning sign would be recall falling while the share of flagged parcels stays the same, because that would mean the reasons behind delays have changed, not just the parcel mix.

## Limitations

1. **The data is synthetic.** All rows come from `generate_data.py`, where delays are built from simple rules plus random noise. Real Delhivery data would be messier, so these results would not carry over directly.
2. **Very few late parcels to test on.** The test set has only 41 late parcels and validation has 42. One or two parcels can change recall a lot, so the test recall of 0.7561 and validation recall of 0.8333 are not very exact.
3. **The threshold and settings come from one validation split.** I picked the 0.5 threshold and the default forest from a single set of 180 rows, where the three tuning options were only a few parcels apart. A different split might give a different choice.
4. **Hub segments are based on 50 parcels per hub.** That is not many, so a few late parcels can move a hub's delay rate a lot. HUB15 Kochi (0.36) has the lowest delay rate in the Escalate group and could move to another segment with new data.
5. **Unclear which regression model to ship.** Linear regression won on validation, but gradient boosting won in cross-validation, and the gaps were small. I chose gradient boosting, but this is a judgement call.
6. **The tier label is used as a model input.** Four hubs have a misleading tier, so the classifier's predictions for them use wrong information until the labels are fixed.
7. **My own experience.** I am new to machine learning and used only the methods from my course (linear and logistic regression, decision trees, random forest, gradient boosting, K-Means). Someone more experienced might try other models or a more careful way to pick the threshold, for example using cross-validation.
