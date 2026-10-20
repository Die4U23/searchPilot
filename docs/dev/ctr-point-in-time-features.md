# Point-in-time CTR training features

Each training sample reads counts from positive training events strictly earlier than its shown_at timestamp. All samples at one timestamp share a frozen prefix; current and same-time labels are applied only after the batch's features are computed. Sampling does not discard positive history events, and output order matches selected rows. Invalid timestamps, nonbinary labels, missing required data, duplicate/invalid sample positions and impressions spanning timestamps are rejected.

Histories start empty at the observation-window boundary. This replaces the old final user_history.parquet backfill and intentionally measures observed positive events, not a user's complete lifetime history. End-of-training tables are frozen for val/test and online scoring. All train times must precede dev times; no dev label contributes to counts. This does not solve position bias, missing pre-window history or historical item-metadata availability.

The policy is recorded as strictly-earlier-positive-events-v1 in new metadata/reports. Existing metrics are historical and must not be presented as results of this policy. No full-data training was run for this patch and improved AUC/ECE is not asserted.

Seven helper tests cover same-time isolation, label perturbation, sample order, unsorted input, unsampled positives, empty samples and invalid inputs. Two loader tests compile the actual data-loading functions from train_ctr.py and use an in-memory Parquet-reader fixture; they exercise the real feature/split functions, not Torch or actual Parquet files. Full training and end-to-end API CI remain separate checks.

This patch follows the versioned artifact bundle change (issues #43/#45); it must not be applied ahead of that dependency without rebasing and retesting.
