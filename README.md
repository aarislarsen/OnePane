Multi-level screenshot similarity grouper with enhanced algorithms.

This tool groups similar screenshots using perceptual hashing and computer vision techniques.
It supports two detection modes:
1. Exact duplicates: Near-identical images (default threshold: 97%)
2. Structural similarity: Same layout/UI with different text (default threshold: 97%)

The script generates a web interface for browsing grouped screenshots with auto-scroll
capabilities and customizable similarity thresholds. Results are heavily cached for
performance across multiple runs.

Key features:
- Multiple hash types: average, perceptual, wavelet, edge-based, color layout
- Persistent caching of hash computations and groupings
- Web interface with customizable view options and auto-scroll
- Editable group names that persist across sessions
