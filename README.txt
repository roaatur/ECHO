latest: 
- models: nano@320, medium@640 
- target: fn (miss count from nano@320)
- horizon: 3 frames ahead
- context window: 20 frames input
- 15 features:
strip-based (3 per strip × 3 strips = 9):
top/mid/bot_laplacian_variance — sharpness
top/mid/bot_small_component_count — small edge blob count
top/mid/bot_gradient_magnitude_std — texture variation
global (6):
global_total_small_components — scene busyness (best feature)
global_overall_edge_density — how many edges overall
global_FAST_keypoint_count — corner/interest points
global_mean_brightness — lighting
global_dark_channel_mean — weather/fog proxy
global_temporal_diff_mean — motion between frames
- per-sequence R² = 0.17. almost flat.
- intra-sequence R² = 0.200: 20% of within-sequence variance explained.
- raw R² = 0.791. misleading.
