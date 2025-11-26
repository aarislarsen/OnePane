#!/usr/bin/env python3
"""
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
"""

import os
import sys
import json
from pathlib import Path
import base64
from io import BytesIO
import time

# Auto-install required dependencies if missing
try:
    from PIL import Image, ImageFilter, ImageEnhance
    import imagehash
    from flask import Flask, render_template_string, send_file, request, jsonify
    import numpy as np
    from scipy.spatial.distance import cosine
    import requests
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install Pillow imagehash flask numpy scipy requests --break-system-packages")
    from PIL import Image, ImageFilter, ImageEnhance
    import imagehash
    from flask import Flask, render_template_string, send_file, request, jsonify
    import numpy as np
    from scipy.spatial.distance import cosine
    import requests

# Cache file definitions - these store computed data to speed up subsequent runs
CACHE_FILE = ".image_hashes_multi.json"          # Perceptual hashes for each image
GROUPS_FILE = ".group_names.json"                # User-defined group names
GROUPINGS_CACHE_FILE = ".groupings_cache.json"   # Pre-computed similarity groupings
WEB_CACHE_FILE = ".web_cache.json"               # Base64-encoded thumbnails for web UI
AI_RESPONSES_FILE = ".ai_responses.json"         # Cached AI analysis responses
CUSTOM_PROMPT_FILE = ".custom_prompt.txt"        # User-defined AI prompt

def extract_edge_structure(img):
    """
    Extract edge-based structural features using Sobel-like filtering.
    This method is highly resistant to text content changes.
    
    Process:
    1. Convert to grayscale to focus on structure
    2. Apply edge detection to find boundaries and shapes
    3. Blur edges to reduce noise and focus on major structures
    4. Generate hash from processed edges
    
    Args:
        img: PIL Image object
        
    Returns:
        imagehash.ImageHash: 12x12 hash representing edge structure
    """
    # Convert to grayscale - color doesn't matter for structural analysis
    gray = img.convert('L')
    
    # Find edges using built-in edge detection filter
    edges = gray.filter(ImageFilter.FIND_EDGES)
    
    # Blur to merge nearby edges and reduce fine detail sensitivity
    edges_blurred = edges.filter(ImageFilter.GaussianBlur(radius=4))
    
    # Generate 12x12 hash (144 bits) for detailed edge structure comparison
    edge_hash = imagehash.average_hash(edges_blurred, hash_size=12)
    return edge_hash

def extract_color_layout(img):
    """
    Extract color distribution in grid layout.
    Resistant to text but sensitive to UI element positions and colors.
    
    This creates a low-resolution color "fingerprint" of the image that captures
    the overall color distribution and layout without being affected by text.
    
    Args:
        img: PIL Image object
        
    Returns:
        numpy.ndarray: Flattened array of normalized RGB values
    """
    # Downscale to 16x16 to get rough color distribution
    img_small = img.resize((16, 16), Image.Resampling.LANCZOS)
    
    # Blur to merge nearby pixels and reduce sensitivity to exact positions
    img_blurred = img_small.filter(ImageFilter.GaussianBlur(radius=2))
    
    # Convert to numpy array for mathematical operations
    arr = np.array(img_blurred)
    
    # Flatten to 1D array (handle both grayscale and RGB)
    if len(arr.shape) == 3:
        arr = arr.reshape(-1)  # RGB: flatten all dimensions
    else:
        arr = arr.flatten()    # Grayscale: simple flatten
    
    # Normalize to 0-1 range for consistent comparison
    return arr.astype(np.float32) / 255.0

def compute_multi_hashes(img):
    """
    Compute multiple hash types for different similarity detection strategies.
    
    Each hash type captures different aspects of image similarity:
    - ahash: Simple average, good for exact duplicates
    - phash: Frequency-based, resistant to minor changes
    - whash: Wavelet-based, good for structural similarity
    - blur_hash: Heavy blur, ignores fine details like text
    - color_hash: Color distribution, layout-aware
    - edge_hash: Edge structure, text-resistant
    - color_layout: Color grid fingerprint
    
    Args:
        img: PIL Image object
        
    Returns:
        dict: Dictionary containing all computed hashes
    """
    # Standard perceptual hashes with varying sensitivities
    ahash = imagehash.average_hash(img, hash_size=8)   # 8x8 = 64 bits
    phash = imagehash.phash(img, hash_size=12)         # 12x12 = 144 bits (more detail)
    whash = imagehash.whash(img, hash_size=8)          # Wavelet hash
    
    # Heavy blur to eliminate text while preserving layout
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=7))
    blur_hash = imagehash.average_hash(img_blurred, hash_size=10)
    
    # Color-focused hash: downscale and blur to get color distribution
    img_small = img.resize((32, 32), Image.Resampling.LANCZOS)
    img_small_blur = img_small.filter(ImageFilter.GaussianBlur(radius=4))
    color_hash = imagehash.average_hash(img_small_blur, hash_size=8)
    
    # Extract edge-based and color layout features
    edge_hash = extract_edge_structure(img)
    color_layout = extract_color_layout(img)
    
    # Encode color layout as base64 string for JSON serialization
    color_layout_str = base64.b64encode(color_layout.tobytes()).decode('ascii')
    
    return {
        'ahash': ahash,
        'phash': phash,
        'whash': whash,
        'blur_hash': blur_hash,
        'color_hash': color_hash,
        'edge_hash': edge_hash,
        'color_layout': color_layout_str
    }

def calculate_similarity(hash1, hash2, hash_type='phash'):
    """
    Calculate similarity percentage between two hashes.
    
    Hamming distance is converted to percentage where:
    - 0% = completely different
    - 100% = identical
    
    Args:
        hash1: First hash dictionary
        hash2: Second hash dictionary
        hash_type: Which hash type to compare (default: 'phash')
        
    Returns:
        float: Similarity percentage (0-100)
    """
    h1 = hash1[hash_type]
    h2 = hash2[hash_type]
    
    # Maximum possible difference (all bits different)
    max_diff = len(h1.hash) ** 2
    
    # Actual Hamming distance (number of different bits)
    actual_diff = h1 - h2
    
    # Convert to percentage similarity
    similarity = (1 - (actual_diff / max_diff)) * 100
    return similarity

def calculate_color_layout_similarity(layout1_str, layout2_str):
    """
    Calculate cosine similarity between color layout vectors.
    
    Cosine similarity measures the angle between two vectors, ranging from:
    - 0% = orthogonal (completely different)
    - 100% = parallel (identical direction/pattern)
    
    Args:
        layout1_str: Base64-encoded color layout
        layout2_str: Base64-encoded color layout
        
    Returns:
        float: Similarity percentage (0-100)
    """
    try:
        # Decode base64 strings back to numpy arrays
        layout1 = np.frombuffer(base64.b64decode(layout1_str), dtype=np.float32)
        layout2 = np.frombuffer(base64.b64decode(layout2_str), dtype=np.float32)
        
        # Calculate cosine similarity (1 - cosine distance)
        similarity = (1 - cosine(layout1, layout2)) * 100
        
        # Ensure non-negative result (can be slightly negative due to floating point)
        return max(0, similarity)
    except:
        # Return 0 if decoding or calculation fails
        return 0

def calculate_composite_similarity(hash1, hash2):
    """
    Calculate weighted similarity across multiple hash types.
    
    This composite score is optimized for detecting screenshots with the same
    layout but different text content. Weights are tuned based on empirical
    testing to prioritize structural features over fine details.
    
    Weight rationale:
    - edge_hash (30%): Strongest indicator of layout similarity
    - blur_hash (25%): Eliminates text, focuses on major elements
    - phash (20%): General perceptual similarity
    - color_hash (15%): Color distribution matters but less than structure
    - whash (10%): Wavelet provides additional structural info
    - color_layout (10%): Fine-grained color positioning
    
    Args:
        hash1: First hash dictionary
        hash2: Second hash dictionary
        
    Returns:
        float: Composite similarity score (0-100)
    """
    # Define weights for each hash type (must sum to ~1.0)
    weights = {
        'edge_hash': 0.30,    # Highest weight: best for structural similarity
        'blur_hash': 0.25,    # High weight: eliminates text effectively
        'phash': 0.20,        # Medium weight: good general measure
        'color_hash': 0.15,   # Lower weight: useful but not critical
        'whash': 0.10         # Lowest weight: supplementary information
    }
    
    # Calculate weighted sum of hash similarities
    total_similarity = 0
    for hash_type, weight in weights.items():
        sim = calculate_similarity(hash1, hash2, hash_type)
        total_similarity += sim * weight
    
    # Add color layout similarity with 10% weight
    color_sim = calculate_color_layout_similarity(
        hash1['color_layout'], 
        hash2['color_layout']
    )
    total_similarity += color_sim * 0.10
    
    return total_similarity

def load_cache(cache_path):
    """
    Load cached image hashes from disk.
    
    Cache structure:
    {
        "image_path": {
            "hashes": {...},
            "filename": "...",
            "mtime": timestamp
        }
    }
    
    Args:
        cache_path: Path to cache JSON file
        
    Returns:
        dict: Cache dictionary (empty if load fails)
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
                
                # Convert hex strings back to ImageHash objects
                for key in cache:
                    for hash_type in ['ahash', 'phash', 'whash', 'blur_hash', 'color_hash', 'edge_hash']:
                        if hash_type in cache[key]['hashes']:
                            cache[key]['hashes'][hash_type] = imagehash.hex_to_hash(
                                cache[key]['hashes'][hash_type]
                            )
                return cache
        except Exception as e:
            print(f"\nWarning: Could not load cache: {e}")
    return {}

def save_cache(cache, cache_path):
    """
    Save image hashes to cache file for future runs.
    
    ImageHash objects are converted to hex strings for JSON serialization.
    
    Args:
        cache: Cache dictionary to save
        cache_path: Path to cache JSON file
    """
    cache_serializable = {}
    
    # Convert ImageHash objects to strings
    for key, value in cache.items():
        hashes_serializable = {}
        for hash_type, hash_val in value['hashes'].items():
            if hash_type == 'color_layout':
                # Color layout is already a string
                hashes_serializable[hash_type] = hash_val
            else:
                # Convert ImageHash to hex string
                hashes_serializable[hash_type] = str(hash_val)
        
        cache_serializable[key] = {
            'hashes': hashes_serializable,
            'filename': value['filename'],
            'mtime': value['mtime']
        }
    
    # Write to disk
    with open(cache_path, 'w') as f:
        json.dump(cache_serializable, f, indent=2)

def load_group_names(groups_path):
    """
    Load saved group names from disk.
    
    Group names are keyed by seed filename and persist across sessions.
    
    Args:
        groups_path: Path to group names JSON file
        
    Returns:
        dict: Group names dictionary (empty if load fails)
    """
    if os.path.exists(groups_path):
        try:
            with open(groups_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_group_names(group_names, groups_path):
    """
    Save group names to disk.
    
    Args:
        group_names: Dictionary of group names
        groups_path: Path to group names JSON file
    """
    with open(groups_path, 'w') as f:
        json.dump(group_names, f, indent=2)

def load_groupings_cache(cache_path):
    """
    Load cached groupings from disk.
    
    Groupings cache stores the complete similarity analysis results to avoid
    recomputing when no images have changed.
    
    Args:
        cache_path: Path to groupings cache JSON file
        
    Returns:
        dict or None: Cached groupings or None if unavailable
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def save_groupings_cache(groups, image_files, exact_threshold, structural_threshold, cache_path):
    """
    Save groupings cache to disk.
    
    Includes threshold values to detect when parameters change and
    invalidate the cache.
    
    Args:
        groups: List of image groups
        image_files: List of image file paths
        exact_threshold: Exact duplicate threshold
        structural_threshold: Structural similarity threshold
        cache_path: Path to groupings cache JSON file
    """
    cache_data = {
        'groups': groups,
        'image_files': [str(f) for f in image_files],
        'exact_threshold': exact_threshold,
        'structural_threshold': structural_threshold
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

def load_web_cache(cache_path):
    """
    Load cached web interface data (base64-encoded thumbnails).
    
    Args:
        cache_path: Path to web cache JSON file
        
    Returns:
        dict or None: Cached web data or None if unavailable
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def save_web_cache(web_groups, cache_path):
    """
    Save web interface cache to disk.
    
    Args:
        web_groups: List of groups with base64-encoded images
        cache_path: Path to web cache JSON file
    """
    with open(cache_path, 'w') as f:
        json.dump(web_groups, f, indent=2)

def load_ai_responses(cache_path):
    """
    Load cached AI responses from disk.
    
    AI responses are keyed by seed filename and persist across sessions.
    
    Args:
        cache_path: Path to AI responses JSON file
        
    Returns:
        dict: AI responses dictionary (empty if load fails)
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_ai_responses(ai_responses, cache_path):
    """
    Save AI responses to disk.
    
    Args:
        ai_responses: Dictionary of AI responses keyed by seed filename
        cache_path: Path to AI responses JSON file
    """
    with open(cache_path, 'w') as f:
        json.dump(ai_responses, f, indent=2)

def load_custom_prompt(prompt_path):
    """
    Load custom AI prompt from disk.
    
    Args:
        prompt_path: Path to custom prompt text file
        
    Returns:
        str: Custom prompt or empty string if not found
    """
    if os.path.exists(prompt_path):
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except:
            pass
    return ""

def save_custom_prompt(prompt, prompt_path):
    """
    Save custom AI prompt to disk.
    
    Args:
        prompt: The custom prompt text
        prompt_path: Path to custom prompt text file
    """
    with open(prompt_path, 'w') as f:
        f.write(prompt)

def group_similar_images(folder_path, exact_threshold=97, structural_threshold=97, cache_path=None):
    """
    Group images by similarity with two-tier thresholds.
    
    This is the main analysis function. It:
    1. Scans for image files
    2. Checks if cached groupings can be reused
    3. Loads or computes perceptual hashes
    4. Groups images based on similarity thresholds
    5. Caches results for future runs
    
    Two-tier threshold system:
    - exact_threshold: For near-identical images (uses ahash)
    - structural_threshold: For same layout/different text (uses composite score)
    
    Args:
        folder_path: Directory containing images
        exact_threshold: Threshold for exact duplicates (default: 97)
        structural_threshold: Threshold for structural similarity (default: 97)
        cache_path: Optional custom cache path
        
    Returns:
        list: List of groups, each group is a list of image info dictionaries
    """
    total_start = time.time()
    
    # Set default cache path if not provided
    if cache_path is None:
        cache_path = os.path.join(folder_path, CACHE_FILE)
    
    groupings_cache_path = os.path.join(folder_path, GROUPINGS_CACHE_FILE)
    
    # ===== STEP 1: Scan for image files =====
    scan_start = time.time()
    image_files = []
    
    # Search for all common image formats (case-insensitive)
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.webp']:
        image_files.extend(Path(folder_path).glob(ext))
        image_files.extend(Path(folder_path).glob(ext.upper()))
    
    scan_time = time.time() - scan_start
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return []
    
    print(f"Found {len(image_files)} images (scan time: {scan_time:.2f}s)")
    
    # ===== STEP 2: Check for cached groupings =====
    cached_groupings = load_groupings_cache(groupings_cache_path)
    if cached_groupings:
        # Convert to sets for efficient comparison
        cached_files = set(cached_groupings['image_files'])
        current_files = set(str(f.absolute()) for f in image_files)
        
        # Check if thresholds match (cache is invalid if thresholds changed)
        thresholds_match = (
            cached_groupings['exact_threshold'] == exact_threshold and
            cached_groupings['structural_threshold'] == structural_threshold
        )
        
        # If file list and thresholds unchanged, use cached groupings
        if cached_files == current_files and thresholds_match:
            print("Using cached groupings (no new images detected)")
            groups = cached_groupings['groups']
            
            # Convert path strings back to Path objects
            for group in groups:
                for img_info in group:
                    img_info['path'] = Path(img_info['path'])
            
            total_time = time.time() - total_start
            print(f"Groupings loaded in {total_time:.2f}s")
            return groups
        else:
            # Cache is invalid, explain why
            if not thresholds_match:
                print("Thresholds changed, recomputing groupings...")
            else:
                print("New or removed images detected, recomputing groupings...")
    
    # ===== STEP 3: Load hash cache =====
    cache_start = time.time()
    cache = load_cache(cache_path)
    cache_time = time.time() - cache_start
    print(f"Cache load time: {cache_time:.2f}s")
    
    # ===== STEP 4: Calculate perceptual hashes =====
    hash_start = time.time()
    hashes = {}
    new_count = 0     # Images processed fresh
    cached_count = 0  # Images loaded from cache
    
    for idx, img_path in enumerate(image_files, 1):
        img_path_str = str(img_path.absolute())
        img_mtime = os.path.getmtime(img_path)  # Modification time
        
        # Check if hash is cached and file hasn't been modified
        if img_path_str in cache and cache[img_path_str]['mtime'] == img_mtime:
            # Use cached hash
            hashes[img_path_str] = {
                'hashes': cache[img_path_str]['hashes'],
                'filename': cache[img_path_str]['filename']
            }
            cached_count += 1
        else:
            # Compute new hash
            try:
                img = Image.open(img_path).convert('RGB')
                img_hashes = compute_multi_hashes(img)
                
                hashes[img_path_str] = {
                    'hashes': img_hashes,
                    'filename': img_path.name
                }
                
                # Update cache
                cache[img_path_str] = {
                    'hashes': img_hashes,
                    'filename': img_path.name,
                    'mtime': img_mtime
                }
                new_count += 1
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                continue
        
        # Progress indicator
        print(f"\rProcessing: {idx}/{len(image_files)} (new: {new_count}, cached: {cached_count})", end='', flush=True)
    
    print()
    hash_time = time.time() - hash_start
    print(f"Hash computation time: {hash_time:.2f}s ({new_count} new, {cached_count} cached)")
    
    # ===== STEP 5: Save updated cache =====
    save_start = time.time()
    save_cache(cache, cache_path)
    save_time = time.time() - save_start
    print(f"Cache save time: {save_time:.2f}s")
    
    # ===== STEP 6: Group similar images =====
    group_start = time.time()
    print("Grouping similar images...", end='', flush=True)
    groups = []
    processed = set()  # Track which images have been assigned to groups
    
    items = list(hashes.items())
    comparison_count = 0
    total_comparisons = len(items) * (len(items) - 1) // 2  # n choose 2
    
    # For each image, find all similar images
    for i, (img1_path, img1_data) in enumerate(items):
        if img1_path in processed:
            continue  # Skip if already grouped
        
        # Create new group with this image as seed
        group = [{
            'path': img1_path,
            'filename': img1_data['filename'],
            'is_seed': True  # First image in group is the seed
        }]
        processed.add(img1_path)
        
        # Compare with all remaining images
        for img2_path, img2_data in items[i+1:]:
            if img2_path in processed:
                continue  # Skip if already grouped
            
            comparison_count += 1
            
            # Progress indicator every 100 comparisons
            if comparison_count % 100 == 0:
                print(f"\rGrouping: {comparison_count}/{total_comparisons}", end='', flush=True)
            
            # Calculate both similarity types
            exact_sim = calculate_similarity(img1_data['hashes'], img2_data['hashes'], 'ahash')
            structural_sim = calculate_composite_similarity(img1_data['hashes'], img2_data['hashes'])
            
            # Add to group if either threshold is met
            if exact_sim >= exact_threshold or structural_sim >= structural_threshold:
                group.append({
                    'path': img2_path,
                    'filename': img2_data['filename'],
                    'is_seed': False,
                    'exact_sim': exact_sim,
                    'structural_sim': structural_sim
                })
                processed.add(img2_path)
        
        groups.append(group)
    
    print()
    group_time = time.time() - group_start
    print(f"Grouping time: {group_time:.2f}s ({len(groups)} groups found)")
    
    # ===== STEP 7: Sort groups and images =====
    # Sort groups by size (smallest first for easier review)
    groups.sort(key=lambda g: len(g))
    
    # Within each group, put seed first, then sort others by similarity
    for group in groups:
        seed = [img for img in group if img.get('is_seed')]
        others = [img for img in group if not img.get('is_seed')]
        others.sort(key=lambda x: x.get('structural_sim', 0), reverse=True)
        group[:] = seed + others
    
    # ===== STEP 8: Save groupings cache =====
    save_groupings_cache(groups, image_files, exact_threshold, structural_threshold, groupings_cache_path)
    print(f"Groupings cache saved")
    
    # ===== Summary =====
    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"TOTAL PROCESSING TIME: {total_time:.2f}s")
    print(f"{'='*50}\n")
    
    return groups

def create_web_interface(groups, folder_path, port=5000):
    """
    Create Flask web interface to display grouped images.
    
    The web interface provides:
    - Thumbnail view of all groups
    - Manual and automatic scrolling controls
    - Adjustable image size and scroll speed
    - Editable group names
    - Lightbox view for full-size images
    - Similarity scores displayed for each image
    
    Args:
        groups: List of image groups
        folder_path: Base folder path for storing cache files
        port: Web server port (default: 5000)
    """
    prep_start = time.time()
    
    app = Flask(__name__)
    
    # ===== Build image path lookup =====
    # Maps filename -> full path for serving images
    image_paths = {}
    for group in groups:
        for img_info in group:
            image_paths[img_info['filename']] = img_info['path']
    
    # ===== Load group names =====
    groups_path = os.path.join(folder_path, GROUPS_FILE)
    group_names = load_group_names(groups_path)
    
    # Initialize default names for groups without saved names
    for idx, group in enumerate(groups):
        seed_filename = group[0]['filename']
        if seed_filename not in group_names:
            group_names[seed_filename] = f"Group {idx + 1}"
    
    # ===== Load AI responses cache =====
    ai_responses_path = os.path.join(folder_path, AI_RESPONSES_FILE)
    ai_responses = load_ai_responses(ai_responses_path)
    
    # ===== Load custom prompt =====
    custom_prompt_path = os.path.join(folder_path, CUSTOM_PROMPT_FILE)
    custom_prompt = load_custom_prompt(custom_prompt_path)
    
    web_cache_path = os.path.join(folder_path, WEB_CACHE_FILE)
    
    # ===== Check for cached web data =====
    cached_web = load_web_cache(web_cache_path)
    if cached_web:
        # Verify cache is still valid (all images exist)
        cache_valid = True
        for group in cached_web:
            for img_info in group:
                if img_info['filename'] not in image_paths:
                    cache_valid = False
                    break
            if not cache_valid:
                break
        
        # Use cache if valid and group count matches
        if cache_valid and len(cached_web) == len(groups):
            print("Using cached web data")
            web_groups = cached_web
            prep_time = time.time() - prep_start
            print(f"Web interface preparation time: {prep_time:.2f}s\n")
        else:
            print("Web cache invalid, regenerating...")
            cached_web = None
    
    # ===== Generate web data if cache unavailable =====
    if not cached_web:
        print("Preparing web interface...", end='', flush=True)
        web_groups = []
        total_images = sum(len(group) for group in groups)
        processed_images = 0
        
        for group in groups:
            web_group = []
            seed_filename = group[0]['filename']
            
            # Convert each image to base64 thumbnail
            for img_info in group:
                try:
                    # Load and resize image
                    img = Image.open(img_info['path'])
                    img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
                    
                    # Encode as PNG base64
                    buffered = BytesIO()
                    img.save(buffered, format="PNG", optimize=True)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Build web image info
                    web_img = {
                        'filename': img_info['filename'],
                        'data': img_str,
                        'is_seed': img_info.get('is_seed', False),
                        'group_name': group_names.get(seed_filename, f"Group {len(web_groups) + 1}")
                    }
                    
                    # Add similarity scores for non-seed images
                    if not img_info.get('is_seed'):
                        web_img['exact_sim'] = img_info.get('exact_sim', 0)
                        web_img['structural_sim'] = img_info.get('structural_sim', 0)
                    
                    web_group.append(web_img)
                    processed_images += 1
                    print(f"\rPreparing web interface: {processed_images}/{total_images}", end='', flush=True)
                except Exception as e:
                    print(f"\nError encoding {img_info['filename']}: {e}")
            
            if web_group:
                web_groups.append(web_group)
        
        print()
        
        # Save web cache for next time
        save_web_cache(web_groups, web_cache_path)
        print(f"Web cache saved")
        
        prep_time = time.time() - prep_start
        print(f"Web interface preparation time: {prep_time:.2f}s\n")
    
    # ===== Calculate statistics =====
    total_images = sum(len(group) for group in web_groups)
    multi_image_groups = sum(1 for group in web_groups if len(group) > 1)
    
    # ===== HTML Template =====
    # Note: This is a large inline HTML/CSS/JS template for the web interface
    # Key features:
    # - Dark theme with GitHub-inspired styling
    # - Controls for image size and scroll speed
    # - Master play/pause button for auto-scrolling all groups
    # - Per-group play/rewind controls
    # - Editable group names
    # - Lightbox for full-size image viewing
    # - Similarity scores displayed on each image
    HTML_TEMPLATE = r"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OnePane - Screenshot comparison utility and viewer</title>
        <style>
            * {
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: #0d1117;
                color: #c9d1d9;
                padding-top: 140px;
            }
            
            /* Fixed control bar at top */
            .controls-bar {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                background: #161b22;
                border-bottom: 1px solid #30363d;
                padding: 15px 20px;
                z-index: 100;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
            }
            
            h1 {
                color: #58a6ff;
                margin: 0 0 15px 0;
                font-size: 20px;
            }
            
            .controls-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 10px;
            }
            
            .control-group {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .control-group label {
                min-width: 100px;
                font-size: 13px;
                color: #8b949e;
            }
            
            /* Slider styling */
            .control-group input[type="range"] {
                flex: 1;
                height: 6px;
                background: #30363d;
                border-radius: 3px;
                outline: none;
                -webkit-appearance: none;
            }
            
            .control-group input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 16px;
                height: 16px;
                background: #58a6ff;
                cursor: pointer;
                border-radius: 50%;
            }
            
            .control-group input[type="range"]::-moz-range-thumb {
                width: 16px;
                height: 16px;
                background: #58a6ff;
                cursor: pointer;
                border-radius: 50%;
                border: none;
            }
            
            .control-value {
                min-width: 50px;
                text-align: right;
                font-size: 13px;
                color: #58a6ff;
                font-weight: 600;
            }
            
            /* Master controls (top right) */
            .master-controls {
                position: fixed;
                top: 20px;
                right: 20px;
                display: flex;
                gap: 12px;
                z-index: 101;
            }
            
            .master-reset-btn, .master-play-btn {
                background: #238636;
                border: none;
                color: white;
                border-radius: 50%;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.2s;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            }
            
            .master-reset-btn {
                background: #6e7681;
                width: 29px;
                height: 29px;
                font-size: 14px;
            }
            
            .master-play-btn {
                width: 43px;
                height: 43px;
                font-size: 18px;
            }
            
            .master-reset-btn:hover {
                background: #8b949e;
            }
            
            .master-play-btn:hover {
                background: #2ea043;
            }
            
            .master-play-btn.paused {
                background: #f0883e;
            }
            
            .master-play-btn.paused:hover {
                background: #fb8500;
            }
            
            /* Statistics bar */
            .stats {
                background: #0d1117;
                padding: 8px 0;
                font-size: 12px;
                color: #6e7681;
            }
            
            .stats strong {
                color: #58a6ff;
            }
            
            /* Main content area */
            .content {
                padding: 20px;
            }
            
            /* Individual group container */
            .group {
                margin-bottom: 20px;
                background: #161b22;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid #30363d;
                transition: border-color 0.3s ease;
            }
            
            .group.master-active {
                border: 3px solid #58a6ff;
                box-shadow: 0 0 10px rgba(88, 166, 255, 0.3);
            }
            
            .group-header {
                font-weight: 600;
                margin-bottom: 8px;
                color: #58a6ff;
                font-size: 13px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            /* Editable group name input */
            .group-name {
                background: transparent;
                border: 1px solid transparent;
                color: #58a6ff;
                padding: 4px 8px;
                border-radius: 3px;
                font-size: 13px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .group-name:hover {
                border-color: #30363d;
                background: #0d1117;
            }
            
            .group-name:focus {
                outline: none;
                border-color: #58a6ff;
                background: #0d1117;
            }
            
            /* Per-group scroll controls */
            .scroll-controls {
                display: flex;
                gap: 8px;
            }
            
            .scroll-btn {
                background: #238636;
                border: none;
                color: white;
                width: 32px;
                height: 32px;
                border-radius: 4px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                transition: background 0.2s;
            }
            
            .scroll-btn:hover {
                background: #2ea043;
            }
            
            .scroll-btn:disabled {
                background: #30363d;
                cursor: not-allowed;
                opacity: 0.5;
            }
            
            /* Search button with cached response - green checkmark state */
            .scroll-btn.has-response {
                background: #1f6feb;
            }
            
            .scroll-btn.has-response:hover {
                background: #388bfd;
            }
            
            /* Horizontal scrolling container */
            .image-container {
                display: flex;
                overflow-x: auto;
                gap: 8px;
                padding: 8px 0;
                scroll-behavior: smooth;
            }
            
            .image-container::-webkit-scrollbar {
                height: 8px;
            }
            
            .image-container::-webkit-scrollbar-track {
                background: #0d1117;
                border-radius: 4px;
            }
            
            .image-container::-webkit-scrollbar-thumb {
                background: #30363d;
                border-radius: 4px;
            }
            
            .image-container::-webkit-scrollbar-thumb:hover {
                background: #484f58;
            }
            
            /* Individual image wrapper */
            .image-wrapper {
                flex-shrink: 0;
                text-align: center;
                background: #0d1117;
                padding: 6px;
                border-radius: 4px;
            }
            
            .image-wrapper img {
                border: 2px solid #30363d;
                border-radius: 4px;
                display: block;
                cursor: pointer;
                transition: border-color 0.2s;
                max-height: 800px;
                max-width: 800px;
            }
            
            .image-wrapper img:hover {
                border-color: #58a6ff;
            }
            
            /* Seed image has green border */
            .image-wrapper.seed img {
                border-color: #238636;
                border-width: 2px;
            }
            
            .image-name {
                font-size: 11px;
                margin-top: 4px;
                word-wrap: break-word;
                max-width: 200px;
                color: #58a6ff;
                cursor: pointer;
                text-decoration: underline;
            }
            
            .image-name:hover {
                color: #79c0ff;
            }
            
            .similarity-info {
                font-size: 9px;
                color: #6e7681;
                margin-top: 2px;
            }
            
            .seed-badge {
                background: #238636;
                color: #ffffff;
                padding: 2px 5px;
                border-radius: 3px;
                font-size: 9px;
                margin-top: 3px;
                display: inline-block;
                font-weight: 600;
            }
            
            /* High similarity gets green text */
            .high-sim {
                color: #3fb950;
            }
            
            /* Lightbox for full-size viewing */
            .lightbox {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.95);
                align-items: center;
                justify-content: center;
            }
            
            .lightbox.active {
                display: flex;
            }
            
            .lightbox-content {
                max-width: 98%;
                max-height: 98%;
                object-fit: contain;
            }
            
            .lightbox-close {
                position: absolute;
                top: 20px;
                right: 35px;
                color: #f1f1f1;
                font-size: 40px;
                font-weight: bold;
                cursor: pointer;
            }
            
            .lightbox-close:hover {
                color: #58a6ff;
            }
            
            /* Advanced menu at bottom */
            .advanced-container {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: #161b22;
                border-top: 1px solid #30363d;
                z-index: 99;
            }
            
            .advanced-toggle {
                padding: 8px 20px;
                cursor: pointer;
                font-size: 12px;
                color: #6e7681;
                user-select: none;
                transition: color 0.2s;
            }
            
            .advanced-toggle:hover {
                color: #8b949e;
            }
            
            #advancedArrow {
                display: inline-block;
                transition: transform 0.2s;
                font-size: 10px;
            }
            
            #advancedArrow.open {
                transform: rotate(90deg);
            }
            
            .advanced-content {
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease;
                background: #0d1117;
                border-top: 1px solid #30363d;
            }
            
            .advanced-content.open {
                max-height: 200px;
            }
            
            .advanced-grid {
                display: flex;
                align-items: center;
                gap: 20px;
                padding: 15px 20px;
                flex-wrap: wrap;
            }
            
            .advanced-group {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .advanced-group label {
                font-size: 13px;
                color: #8b949e;
                white-space: nowrap;
            }
            
            .ai-select {
                background: #161b22;
                border: 1px solid #30363d;
                color: #c9d1d9;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 13px;
                cursor: pointer;
                outline: none;
            }
            
            .ai-select:hover {
                border-color: #58a6ff;
            }
            
            .ai-select:focus {
                border-color: #58a6ff;
            }
            
            .api-key-input {
                background: #161b22;
                border: 1px solid #30363d;
                color: #c9d1d9;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 13px;
                outline: none;
                width: 300px;
            }
            
            .api-key-input:hover {
                border-color: #58a6ff;
            }
            
            .api-key-input:focus {
                border-color: #58a6ff;
            }
            
            .api-key-input::placeholder {
                color: #6e7681;
            }
            
            .save-key-btn {
                background: #1f6feb;
                border: none;
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                transition: background 0.2s;
            }
            
            .save-key-btn:hover {
                background: #388bfd;
            }
            
            .test-connection-btn {
                background: #238636;
                border: none;
                color: white;
                padding: 6px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                transition: background 0.2s;
            }
            
            .test-connection-btn:hover {
                background: #2ea043;
            }
            
            .connection-status {
                font-size: 12px;
                margin-left: 10px;
            }
            
            .connection-status.success {
                color: #3fb950;
            }
            
            .connection-status.error {
                color: #f85149;
            }
            
            .connection-status.testing {
                color: #58a6ff;
            }
            
            .define-prompt-btn {
                background: #6e40c9;
                border: none;
                color: white;
                padding: 6px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                transition: background 0.2s;
            }
            
            .define-prompt-btn:hover {
                background: #8957e5;
            }
            
            /* Prompt modal styles */
            .prompt-modal {
                display: none;
                position: fixed;
                z-index: 2000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.85);
                align-items: center;
                justify-content: center;
            }
            
            .prompt-modal.active {
                display: flex;
            }
            
            .prompt-modal-content {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                max-width: 700px;
                width: 90%;
                display: flex;
                flex-direction: column;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            }
            
            .prompt-modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 16px 20px;
                border-bottom: 1px solid #30363d;
            }
            
            .prompt-modal-title {
                color: #58a6ff;
                font-size: 16px;
                font-weight: 600;
            }
            
            .prompt-modal-close {
                color: #8b949e;
                font-size: 28px;
                cursor: pointer;
                line-height: 1;
            }
            
            .prompt-modal-close:hover {
                color: #f85149;
            }
            
            .prompt-modal-body {
                padding: 20px;
            }
            
            .prompt-modal-body label {
                display: block;
                color: #8b949e;
                font-size: 13px;
                margin-bottom: 8px;
            }
            
            .prompt-textarea {
                width: 100%;
                height: 200px;
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 4px;
                color: #c9d1d9;
                padding: 12px;
                font-size: 14px;
                font-family: inherit;
                resize: vertical;
                outline: none;
            }
            
            .prompt-textarea:focus {
                border-color: #58a6ff;
            }
            
            .prompt-modal-footer {
                display: flex;
                justify-content: flex-end;
                gap: 10px;
                padding: 16px 20px;
                border-top: 1px solid #30363d;
            }
            
            .prompt-cancel-btn {
                background: #21262d;
                border: 1px solid #30363d;
                color: #c9d1d9;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                transition: background 0.2s;
            }
            
            .prompt-cancel-btn:hover {
                background: #30363d;
            }
            
            .prompt-save-btn {
                background: #238636;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 13px;
                transition: background 0.2s;
            }
            
            .prompt-save-btn:hover {
                background: #2ea043;
            }
        </style>
    </head>
    <body>
        <!-- Master controls: reset and play/pause all groups -->
        <div class="master-controls">
            <button class="master-reset-btn" id="masterResetBtn" onclick="resetMasterPlay()" title="Reset scroll">&#8634;</button>
            <button class="master-play-btn" id="masterPlayBtn" onclick="toggleMasterPlay()" title="Auto-scroll all groups">&#9654;</button>
        </div>
        
        <!-- Top control bar -->
        <div class="controls-bar">
            <h1>OnePane - Screenshot comparison utility and viewer</h1>
            <div class="controls-grid">
                <div class="control-group">
                    <label>Image Size:</label>
                    <input type="range" id="sizeSlider" min="100" max="800" value="150" step="10">
                    <span class="control-value" id="sizeValue">150px</span>
                </div>
                <div class="control-group">
                    <label>Scroll Speed:</label>
                    <input type="range" id="speedSlider" min="1" max="10" value="5" step="1">
                    <span class="control-value" id="speedValue">5</span>
                </div>
            </div>
            <div class="stats">
                <strong>Total groups:</strong> {{ groups|length }} | 
                <strong>Total images:</strong> {{ total_images }} | 
                <strong>Groups with matches:</strong> {{ multi_image_groups }}
            </div>
        </div>
        
        <!-- Main content: groups of images -->
        <div class="content">
            {% for group in groups %}
            <div class="group" data-group-index="{{ loop.index0 }}" data-seed="{{ group[0].filename }}">
                <div class="group-header">
                    <div class="scroll-controls">
                        <button class="scroll-btn play-btn" onclick="scrollGroup({{ loop.index0 }}, 'play')" title="Auto-scroll">&#9654;</button>
                        <button class="scroll-btn rewind-btn" onclick="scrollGroup({{ loop.index0 }}, 'rewind')" title="Rewind" disabled>&#9198;</button>
                        {% if group[0].filename in ai_responses %}
                        <button class="scroll-btn search-btn has-response" onclick="handleSearchButton({{ loop.index0 }})" title="View AI Analysis">&#128196;</button>
                        {% else %}
                        <button class="scroll-btn search-btn" onclick="handleSearchButton({{ loop.index0 }})" title="Search with AI">&#128269;</button>
                        {% endif %}
                    </div>
                    <input type="text" 
                           class="group-name" 
                           value="{{ group[0].group_name }}" 
                           data-seed="{{ group[0].filename }}"
                           onblur="saveGroupName(this)"
                           onkeypress="if(event.key==='Enter') this.blur()">
                    <span style="color: #6e7681; font-weight: normal;">&#8212; {{ group|length }} image{{ 's' if group|length != 1 else '' }}</span>
                </div>
                <div class="image-container" id="group-{{ loop.index0 }}">
                    {% for img_info in group %}
                    <div class="image-wrapper {% if img_info.is_seed %}seed{% endif %}">
                        <img src="data:image/png;base64,{{ img_info.data }}" 
                             alt="{{ img_info.filename }}" 
                             loading="lazy"
                             class="group-image"
                             data-filename="{{ img_info.filename }}"
                             onclick="openLightbox('{{ img_info.filename }}')">
                        <div class="image-name" onclick="window.open('/image/{{ img_info.filename }}', '_blank')">
                            {{ img_info.filename }}
                        </div>
                        {% if img_info.is_seed %}
                        <div class="seed-badge">SEED</div>
                        {% else %}
                        <div class="similarity-info {% if img_info.structural_sim >= 90 %}high-sim{% endif %}">
                            Struct: {{ "%.1f"|format(img_info.structural_sim) }}%
                            {% if img_info.exact_sim >= 95 %}
                            | Exact: {{ "%.1f"|format(img_info.exact_sim) }}%
                            {% endif %}
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Lightbox overlay for full-size images -->
        <div id="lightbox" class="lightbox" onclick="closeLightbox()">
            <span class="lightbox-close">&times;</span>
            <img class="lightbox-content" id="lightbox-img">
        </div>
        
        <!-- Advanced menu at bottom -->
        <div class="advanced-container">
            <div class="advanced-toggle" onclick="toggleAdvanced()">
                <span id="advancedArrow">&#9654;</span> Advanced
            </div>
            <div class="advanced-content" id="advancedContent">
                <div class="advanced-grid">
                    <div class="advanced-group">
                        <label>AI Provider:</label>
                        <select id="aiProvider" class="ai-select" onchange="updateModelDropdown()">
                            <option value="perplexity" selected>Perplexity</option>
                            <option value="claude">Claude</option>
                            <option value="chatgpt">ChatGPT</option>
                        </select>
                    </div>
                    <div class="advanced-group">
                        <label>Model:</label>
                        <select id="aiModel" class="ai-select">
                            <option value="sonar">sonar</option>
                            <option value="sonar-pro">sonar-pro</option>
                            <option value="sonar-reasoning">sonar-reasoning</option>
                            <option value="sonar-reasoning-pro">sonar-reasoning-pro</option>
                        </select>
                    </div>
                    <div class="advanced-group">
                        <label>API Key:</label>
                        <input type="password" id="apiKey" class="api-key-input" placeholder="Enter API key">
                    </div>
                    <div class="advanced-group">
                        <button class="test-connection-btn" onclick="testAIConnection()">Check AI Connection</button>
                        <span id="connectionStatus" class="connection-status"></span>
                    </div>
                    <div class="advanced-group">
                        <button class="define-prompt-btn" onclick="openPromptModal()">Define Prompt</button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Prompt definition modal -->
        <div id="promptModal" class="prompt-modal">
            <div class="prompt-modal-content">
                <div class="prompt-modal-header">
                    <span class="prompt-modal-title">Define AI Prompt</span>
                    <span class="prompt-modal-close" onclick="closePromptModal()">&times;</span>
                </div>
                <div class="prompt-modal-body">
                    <label for="customPrompt">Enter the prompt to send with each screenshot:</label>
                    <textarea id="customPrompt" class="prompt-textarea" placeholder="Describe what you want the AI to analyze or look for in the screenshots...">{{ custom_prompt }}</textarea>
                </div>
                <div class="prompt-modal-footer">
                    <button class="prompt-cancel-btn" onclick="closePromptModal()">Cancel</button>
                    <button class="prompt-save-btn" onclick="saveCustomPrompt()">Save Prompt</button>
                </div>
            </div>
        </div>
        
        <script>
            // ===== Global state variables =====
            let scrollIntervals = {};         // Tracks active scroll intervals per group
            let scrollSpeed = 11;             // Pixels per interval (adjusted by speed slider)
            let verticalScrollDuration = 800; // Duration for viewport scroll animations
            let masterPlayActive = false;     // Whether master auto-scroll is running
            let masterPlayPaused = false;     // Whether master auto-scroll is paused
            let currentMasterRow = 0;         // Current row being scrolled by master play
            
            // ===== Cached AI responses (keyed by seed filename) =====
            let cachedAIResponses = {{ ai_responses | tojson | safe }};
            
            // ===== Image size control =====
            document.getElementById('sizeSlider').addEventListener('input', function() {
                const size = this.value;
                document.getElementById('sizeValue').textContent = size + 'px';
                // Apply new size to all images (set both height and width)
                document.querySelectorAll('.group-image').forEach(img => {
                    img.style.maxHeight = size + 'px';
                    img.style.maxWidth = size + 'px';
                });
            });
            
            // ===== Scroll speed control =====
            document.getElementById('speedSlider').addEventListener('input', function() {
                const speed = parseInt(this.value);
                document.getElementById('speedValue').textContent = speed;
                // Map 1-10 slider to scroll speed (2-20 pixels/interval)
                scrollSpeed = 2 + ((speed - 1) / 9) * 18;
                // Adjust vertical scroll duration inversely (faster scroll = shorter duration)
                verticalScrollDuration = 1400 - ((speed - 1) / 9) * 1200;
            });
            
            // ===== Scroll capability detection =====
            function canScroll(groupIndex) {
                const container = document.getElementById('group-' + groupIndex);
                // Check if content width exceeds container width (scrollable)
                return container.scrollWidth > container.clientWidth;
            }
            
            // ===== Per-group scroll control =====
            function scrollGroup(groupIndex, action) {
                const container = document.getElementById('group-' + groupIndex);
                const playBtn = document.querySelectorAll('.play-btn')[groupIndex];
                const rewindBtn = document.querySelectorAll('.rewind-btn')[groupIndex];
                
                if (action === 'play') {
                    // Toggle play/pause
                    if (scrollIntervals[groupIndex]) {
                        // Currently scrolling - pause it
                        clearInterval(scrollIntervals[groupIndex]);
                        scrollIntervals[groupIndex] = null;
                        playBtn.innerHTML = '&#9654;';
                        playBtn.title = 'Auto-scroll';
                    } else {
                        // Start scrolling
                        if (!canScroll(groupIndex)) {
                            // No scrollable content - wait based on scroll speed then notify completion
                            const waitTime = 2000 - ((scrollSpeed - 2) / 18) * 1500; // 2000ms at slowest, 500ms at fastest
                            setTimeout(() => {
                                if (masterPlayActive && !masterPlayPaused && currentMasterRow === groupIndex) {
                                    onMasterRowComplete(groupIndex);
                                }
                            }, waitTime);
                            return;
                        }
                        
                        // Start scroll interval
                        scrollIntervals[groupIndex] = setInterval(() => {
                            container.scrollLeft += scrollSpeed;
                            
                            // Check if reached end
                            if (container.scrollLeft >= container.scrollWidth - container.clientWidth) {
                                clearInterval(scrollIntervals[groupIndex]);
                                scrollIntervals[groupIndex] = null;
                                playBtn.innerHTML = '&#9654;';
                                playBtn.title = 'Auto-scroll';
                                rewindBtn.disabled = false;
                                
                                // Notify master play that this row completed
                                if (masterPlayActive && !masterPlayPaused && currentMasterRow === groupIndex) {
                                    onMasterRowComplete(groupIndex);
                                }
                            }
                        }, 20);  // Update every 20ms
                        playBtn.innerHTML = '&#10073;&#10073;';
                        playBtn.title = 'Pause';
                        rewindBtn.disabled = true;
                    }
                } else if (action === 'rewind') {
                    // Reset scroll position
                    container.scrollLeft = 0;
                    rewindBtn.disabled = true;
                }
            }
            
            // ===== Highlight active master play row =====
            function setActiveMasterRow(groupIndex) {
                // Remove highlight from all rows
                document.querySelectorAll('.group').forEach(group => {
                    group.classList.remove('master-active');
                });
                
                // Add highlight to current row
                if (groupIndex !== null && groupIndex !== undefined) {
                    const activeGroup = document.querySelector(`[data-group-index="${groupIndex}"]`);
                    if (activeGroup) {
                        activeGroup.classList.add('master-active');
                    }
                }
            }
            
            // ===== Master play row completion callback =====
            function onMasterRowComplete(groupIndex) {
                if (!masterPlayActive || masterPlayPaused) return;
                
                // Move to next row
                const totalGroups = document.querySelectorAll('.group').length;
                const nextRow = groupIndex + 1;
                
                if (nextRow >= totalGroups) {
                    // Finished all rows - stop master play
                    stopMasterPlay();
                    return;
                }
                
                // Update current row
                currentMasterRow = nextRow;
                
                // Highlight the next row
                setActiveMasterRow(nextRow);
                
                // Scroll viewport to next row with offset for fixed header
                const groupElement = document.querySelector(`[data-group-index="${nextRow}"]`);
                if (groupElement) {
                    const headerOffset = 160; // Controls bar height (140px) + extra padding (20px)
                    const elementPosition = groupElement.getBoundingClientRect().top;
                    const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
                    
                    window.scrollTo({
                        top: offsetPosition,
                        behavior: 'smooth'
                    });
                    
                    // Wait for viewport scroll, then start scrolling the row
                    setTimeout(() => {
                        if (masterPlayActive && !masterPlayPaused) {
                            const container = document.getElementById('group-' + nextRow);
                            container.scrollLeft = 0; // Reset scroll position
                            scrollGroup(nextRow, 'play');
                        }
                    }, verticalScrollDuration);
                }
            }
            
            // ===== Master play toggle =====
            function toggleMasterPlay() {
                if (!masterPlayActive) {
                    // Not running - start from beginning
                    startMasterPlay();
                } else if (masterPlayPaused) {
                    // Currently paused - resume
                    resumeMasterPlay();
                } else {
                    // Currently running - pause
                    pauseMasterPlay();
                }
            }
            
            // ===== Master play: start =====
            function startMasterPlay() {
                masterPlayActive = true;
                masterPlayPaused = false;
                currentMasterRow = 0;
                
                const masterBtn = document.getElementById('masterPlayBtn');
                masterBtn.innerHTML = '&#10073;&#10073;';
                masterBtn.classList.remove('paused');
                masterBtn.title = 'Pause auto-scroll';
                
                // Highlight first row
                setActiveMasterRow(0);
                
                // Scroll to top
                window.scrollTo({ 
                    top: 0,
                    behavior: 'smooth' 
                });
                
                // Wait for viewport scroll to complete, then start first row
                setTimeout(() => {
                    if (masterPlayActive && !masterPlayPaused) {
                        const container = document.getElementById('group-0');
                        container.scrollLeft = 0; // Reset scroll position
                        scrollGroup(0, 'play');
                    }
                }, 1000);
            }
            
            // ===== Master play: pause =====
            function pauseMasterPlay() {
                masterPlayPaused = true;
                
                const masterBtn = document.getElementById('masterPlayBtn');
                masterBtn.innerHTML = '&#9654;';
                masterBtn.classList.add('paused');
                masterBtn.title = 'Resume auto-scroll';
                
                // Pause the current row's scrolling
                if (scrollIntervals[currentMasterRow]) {
                    clearInterval(scrollIntervals[currentMasterRow]);
                    scrollIntervals[currentMasterRow] = null;
                    const playBtn = document.querySelectorAll('.play-btn')[currentMasterRow];
                    playBtn.innerHTML = '&#9654;';
                    playBtn.title = 'Auto-scroll';
                }
            }
            
            // ===== Master play: resume =====
            function resumeMasterPlay() {
                masterPlayPaused = false;
                
                const masterBtn = document.getElementById('masterPlayBtn');
                masterBtn.innerHTML = '&#10073;&#10073;';
                masterBtn.classList.remove('paused');
                masterBtn.title = 'Pause auto-scroll';
                
                // Resume scrolling the current row from where it left off
                scrollGroup(currentMasterRow, 'play');
            }
            
            // ===== Master play: stop =====
            function stopMasterPlay() {
                masterPlayActive = false;
                masterPlayPaused = false;
                currentMasterRow = 0;
                
                const masterBtn = document.getElementById('masterPlayBtn');
                masterBtn.innerHTML = '&#9654;';
                masterBtn.classList.remove('paused');
                masterBtn.title = 'Auto-scroll all groups';
                
                // Remove highlight from all rows
                setActiveMasterRow(null);
                
                // Stop any active scrolling
                Object.keys(scrollIntervals).forEach(groupIndex => {
                    if (scrollIntervals[groupIndex]) {
                        clearInterval(scrollIntervals[groupIndex]);
                        scrollIntervals[groupIndex] = null;
                        const playBtn = document.querySelectorAll('.play-btn')[groupIndex];
                        playBtn.innerHTML = '&#9654;';
                        playBtn.title = 'Auto-scroll';
                    }
                });
            }
            
            // ===== Master play: reset =====
            function resetMasterPlay() {
                // Stop master play completely
                stopMasterPlay();
                
                // Scroll to top
                window.scrollTo({ 
                    top: 0,
                    behavior: 'smooth' 
                });
                
                // Reset all row scroll positions
                document.querySelectorAll('.image-container').forEach((container, index) => {
                    container.scrollLeft = 0;
                    const rewindBtn = document.querySelectorAll('.rewind-btn')[index];
                    if (rewindBtn) {
                        rewindBtn.disabled = true;
                    }
                });
            }
            
            // ===== Enable rewind button when scrolled =====
            document.querySelectorAll('.image-container').forEach((container, index) => {
                container.addEventListener('scroll', () => {
                    const rewindBtn = document.querySelectorAll('.rewind-btn')[index];
                    if (container.scrollLeft > 0) {
                        rewindBtn.disabled = false;
                    } else {
                        rewindBtn.disabled = true;
                    }
                });
            });
            
            // ===== Lightbox functions =====
            function openLightbox(filename) {
                const lightbox = document.getElementById('lightbox');
                const lightboxImg = document.getElementById('lightbox-img');
                
                lightboxImg.src = '/image/' + filename;
                lightbox.classList.add('active');
            }
            
            function closeLightbox() {
                document.getElementById('lightbox').classList.remove('active');
            }
            
            // Close lightbox on Escape key
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    closeLightbox();
                }
            });
            
            // ===== Save group name via AJAX =====
            function saveGroupName(input) {
                const seedFilename = input.dataset.seed;
                const newName = input.value;
                
                fetch('/rename_group', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        seed: seedFilename,
                        name: newName
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        alert('Failed to save group name');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            }
            
            // ===== Advanced menu toggle =====
            function toggleAdvanced() {
                const content = document.getElementById('advancedContent');
                const arrow = document.getElementById('advancedArrow');
                
                content.classList.toggle('open');
                arrow.classList.toggle('open');
            }
            
            // ===== Model definitions for each provider =====
            const modelsByProvider = {
                'perplexity': [
                    { value: 'sonar', label: 'Sonar (lightweight search)' },
                    { value: 'sonar-pro', label: 'Sonar Pro (advanced search)' },
                    { value: 'sonar-reasoning', label: 'Sonar Reasoning' },
                    { value: 'sonar-reasoning-pro', label: 'Sonar Reasoning Pro' },
                    { value: 'sonar-deep-research', label: 'Sonar Deep Research' }
                ],
                'claude': [
                    { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet' },
                    { value: 'claude-3-5-haiku-20241022', label: 'Claude 3.5 Haiku' },
                    { value: 'claude-3-opus-20240229', label: 'Claude 3 Opus' },
                    { value: 'claude-3-haiku-20240307', label: 'Claude 3 Haiku' }
                ],
                'chatgpt': [
                    { value: 'gpt-4o', label: 'GPT-4o' },
                    { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
                    { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
                    { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' },
                    { value: 'o1', label: 'O1' },
                    { value: 'o1-mini', label: 'O1 Mini' }
                ]
            };
            
            // ===== Update model dropdown based on provider =====
            function updateModelDropdown() {
                const provider = document.getElementById('aiProvider').value;
                const modelSelect = document.getElementById('aiModel');
                
                // Clear existing options
                modelSelect.innerHTML = '';
                
                // Get models for selected provider
                const models = modelsByProvider[provider] || [];
                
                // Populate dropdown
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.value;
                    option.textContent = model.label;
                    modelSelect.appendChild(option);
                });
                
                // Load saved API key for the selected provider
                loadApiKey();
            }
            
            // ===== Save API key =====
            function saveApiKey() {
                const provider = document.getElementById('aiProvider').value;
                const apiKey = document.getElementById('apiKey').value;
                
                if (!apiKey) {
                    alert('Please enter an API key');
                    return;
                }
                
                fetch('/save_api_key', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        provider: provider,
                        api_key: apiKey
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('API key saved successfully');
                    } else {
                        alert('Failed to save API key: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(error => {
                    alert('Error saving API key');
                    console.error('Error:', error);
                });
            }
            
            // ===== Load API key for selected provider =====
            function loadApiKey() {
                const provider = document.getElementById('aiProvider').value;
                
                fetch('/get_api_key', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        provider: provider
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.api_key) {
                        document.getElementById('apiKey').value = data.api_key;
                    } else {
                        document.getElementById('apiKey').value = '';
                    }
                })
                .catch(error => {
                    console.error('Error loading API key:', error);
                    document.getElementById('apiKey').value = '';
                });
            }
            
            // ===== Test AI connection =====
            function testAIConnection() {
                const provider = document.getElementById('aiProvider').value;
                const model = document.getElementById('aiModel').value;
                const apiKey = document.getElementById('apiKey').value;
                const statusEl = document.getElementById('connectionStatus');
                
                if (!apiKey) {
                    statusEl.innerHTML = '&#10007; Please enter an API key';
                    statusEl.className = 'connection-status error';
                    return;
                }
                
                statusEl.textContent = 'Saving and testing...';
                statusEl.className = 'connection-status testing';
                
                // First save the API key
                fetch('/save_api_key', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        provider: provider,
                        api_key: apiKey
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        statusEl.innerHTML = '&#10007; Failed to save API key';
                        statusEl.className = 'connection-status error';
                        return;
                    }
                    
                    // Now test the connection
                    return fetch('/test_ai_connection', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            provider: provider,
                            model: model
                        })
                    });
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        statusEl.innerHTML = '&#10003; Connected';
                        statusEl.className = 'connection-status success';
                    } else {
                        statusEl.innerHTML = '&#10007; ' + (data.error || 'Connection failed');
                        statusEl.className = 'connection-status error';
                    }
                })
                .catch(error => {
                    statusEl.innerHTML = '&#10007; Network error';
                    statusEl.className = 'connection-status error';
                    console.error('Error:', error);
                });
            }
            
            // Load API key on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadApiKey();
                
                // Set up prompt modal close handlers
                const promptModal = document.getElementById('promptModal');
                promptModal.addEventListener('click', function(e) {
                    if (e.target === promptModal) {
                        closePromptModal();
                    }
                });
                
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape' && promptModal.classList.contains('active')) {
                        closePromptModal();
                    }
                });
            });
            
            // ===== Prompt Modal Functions =====
            function openPromptModal() {
                const modal = document.getElementById('promptModal');
                modal.classList.add('active');
                document.getElementById('customPrompt').focus();
            }
            
            function closePromptModal() {
                const modal = document.getElementById('promptModal');
                modal.classList.remove('active');
            }
            
            function saveCustomPrompt() {
                const prompt = document.getElementById('customPrompt').value;
                
                fetch('/save_custom_prompt', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        closePromptModal();
                        // Show brief success message
                        const btn = document.querySelector('.define-prompt-btn');
                        const originalText = btn.textContent;
                        btn.textContent = 'Saved!';
                        btn.style.background = '#238636';
                        setTimeout(() => {
                            btn.textContent = originalText;
                            btn.style.background = '';
                        }, 1500);
                    } else {
                        alert('Failed to save prompt: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error saving prompt');
                });
            }
            
            // ===== Handle search button click =====
            function handleSearchButton(groupIndex) {
                const groupElement = document.querySelector(`[data-group-index="${groupIndex}"]`);
                const seedFilename = groupElement.dataset.seed;
                const groupName = groupElement.querySelector('.group-name').value;
                
                // Check if we have a cached response
                if (cachedAIResponses[seedFilename]) {
                    // Show the cached response immediately
                    showAIResponseModal(groupName, cachedAIResponses[seedFilename]);
                } else {
                    // No cached response, fetch from AI
                    searchGroup(groupIndex);
                }
            }
            
            // ===== Search group with AI (only first/seed image) =====
            async function searchGroup(groupIndex) {
                const provider = document.getElementById('aiProvider').value;
                const model = document.getElementById('aiModel').value;
                
                // Get the group container and collect ONLY the first (seed) image
                const groupElement = document.querySelector(`[data-group-index="${groupIndex}"]`);
                const seedFilename = groupElement.dataset.seed;
                const groupName = groupElement.querySelector('.group-name').value;
                const images = groupElement.querySelectorAll('.group-image');
                
                if (images.length === 0) {
                    alert('No images found in this group.');
                    return;
                }
                
                // Get only the first (seed) image
                const firstImg = images[0];
                const src = firstImg.getAttribute('src');
                
                if (!src || !src.startsWith('data:image/')) {
                    alert('Could not extract image data from this group.');
                    return;
                }
                
                const base64Data = src.split(',')[1];
                const mediaType = src.split(';')[0].split(':')[1];
                const imageData = [{
                    filename: firstImg.dataset.filename,
                    data: base64Data,
                    media_type: mediaType
                }];
                
                // Show loading state on the search button
                const searchBtn = groupElement.querySelector('.search-btn');
                const originalContent = searchBtn.innerHTML;
                const originalClass = searchBtn.className;
                searchBtn.innerHTML = '&#8987;'; // Hourglass
                searchBtn.disabled = true;
                searchBtn.style.background = '#6e7681';
                
                try {
                    const response = await fetch('/search_with_ai', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            provider: provider,
                            model: model,
                            group_name: groupName,
                            seed_filename: seedFilename,
                            images: imageData
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        // Cache the response locally
                        cachedAIResponses[seedFilename] = result.response;
                        
                        // Update button to show "has response" state
                        searchBtn.innerHTML = '&#128196;'; // Document/page icon
                        searchBtn.disabled = false;
                        searchBtn.className = 'scroll-btn search-btn has-response';
                        searchBtn.style.background = '';
                        searchBtn.title = 'View AI Analysis';
                        
                        // Don't show modal automatically - user can click the button to view
                    } else {
                        alert('AI Search Error: ' + (result.error || 'Unknown error'));
                        // Restore button state on error
                        searchBtn.innerHTML = originalContent;
                        searchBtn.disabled = false;
                        searchBtn.className = originalClass;
                        searchBtn.style.background = '';
                    }
                } catch (error) {
                    console.error('Search error:', error);
                    alert('Network error while searching. Please check your connection.');
                    // Restore button state on error
                    searchBtn.innerHTML = originalContent;
                    searchBtn.disabled = false;
                    searchBtn.className = originalClass;
                    searchBtn.style.background = '';
                }
            }
            
            // ===== AI Response Modal =====
            function showAIResponseModal(groupName, response) {
                // Create modal if it doesn't exist
                let modal = document.getElementById('aiResponseModal');
                if (!modal) {
                    modal = document.createElement('div');
                    modal.id = 'aiResponseModal';
                    modal.className = 'ai-modal';
                    modal.innerHTML = `
                        <div class="ai-modal-content">
                            <div class="ai-modal-header">
                                <span class="ai-modal-title"></span>
                                <span class="ai-modal-close" onclick="closeAIResponseModal()">&times;</span>
                            </div>
                            <div class="ai-modal-body"></div>
                        </div>
                    `;
                    document.body.appendChild(modal);
                    
                    // Add modal styles
                    const style = document.createElement('style');
                    style.textContent = `
                        .ai-modal {
                            display: none;
                            position: fixed;
                            z-index: 2000;
                            left: 0;
                            top: 0;
                            width: 100%;
                            height: 100%;
                            background-color: rgba(0, 0, 0, 0.85);
                            align-items: center;
                            justify-content: center;
                        }
                        .ai-modal.active {
                            display: flex;
                        }
                        .ai-modal-content {
                            background: #161b22;
                            border: 1px solid #30363d;
                            border-radius: 8px;
                            max-width: 800px;
                            max-height: 80vh;
                            width: 90%;
                            display: flex;
                            flex-direction: column;
                            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
                        }
                        .ai-modal-header {
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                            padding: 16px 20px;
                            border-bottom: 1px solid #30363d;
                        }
                        .ai-modal-title {
                            color: #58a6ff;
                            font-size: 16px;
                            font-weight: 600;
                        }
                        .ai-modal-close {
                            color: #8b949e;
                            font-size: 28px;
                            cursor: pointer;
                            line-height: 1;
                        }
                        .ai-modal-close:hover {
                            color: #f85149;
                        }
                        .ai-modal-body {
                            padding: 20px;
                            overflow-y: auto;
                            color: #c9d1d9;
                            font-size: 14px;
                            line-height: 1.6;
                        }
                        .ai-modal-body .info-table {
                            width: 100%;
                            border-collapse: collapse;
                            margin-bottom: 16px;
                        }
                        .ai-modal-body .info-table th {
                            text-align: left;
                            padding: 8px 12px;
                            background: #0d1117;
                            border: 1px solid #30363d;
                            color: #8b949e;
                            font-weight: 500;
                            width: 35%;
                        }
                        .ai-modal-body .info-table td {
                            padding: 8px 12px;
                            background: #161b22;
                            border: 1px solid #30363d;
                            color: #c9d1d9;
                        }
                        .ai-modal-body .section-divider {
                            border: none;
                            border-top: 1px solid #30363d;
                            margin: 20px 0;
                        }
                        .ai-modal-body .section-title {
                            color: #58a6ff;
                            font-size: 15px;
                            font-weight: 600;
                            margin: 16px 0 10px 0;
                        }
                        .ai-modal-body .section-content {
                            color: #c9d1d9;
                            line-height: 1.7;
                            margin-bottom: 8px;
                        }
                        .ai-modal-body .creds-table {
                            width: 100%;
                            border-collapse: collapse;
                            margin: 10px 0;
                            font-size: 13px;
                        }
                        .ai-modal-body .creds-table th {
                            text-align: left;
                            padding: 8px 10px;
                            background: #0d1117;
                            border: 1px solid #30363d;
                            color: #8b949e;
                            font-weight: 500;
                        }
                        .ai-modal-body .creds-table td {
                            padding: 8px 10px;
                            background: #161b22;
                            border: 1px solid #30363d;
                            color: #c9d1d9;
                            font-family: monospace;
                        }
                        .ai-modal-body .creds-table td.notes {
                            font-family: inherit;
                            font-size: 12px;
                            color: #8b949e;
                        }
                        .ai-modal-body .highlight {
                            color: #f0f6fc;
                            font-weight: 500;
                        }
                        .ai-modal-body .credential {
                            background: #0d1117;
                            padding: 2px 6px;
                            border-radius: 3px;
                            font-family: monospace;
                        }
                    `;
                    document.head.appendChild(style);
                    
                    // Close on click outside
                    modal.addEventListener('click', function(e) {
                        if (e.target === modal) {
                            closeAIResponseModal();
                        }
                    });
                    
                    // Close on Escape key
                    document.addEventListener('keydown', function(e) {
                        if (e.key === 'Escape' && modal.classList.contains('active')) {
                            closeAIResponseModal();
                        }
                    });
                }
                
                // Update modal content with formatted response
                modal.querySelector('.ai-modal-title').textContent = 'AI Analysis: ' + groupName;
                modal.querySelector('.ai-modal-body').innerHTML = formatAIResponse(response);
                modal.classList.add('active');
            }
            
            // ===== Format AI response with tables and sections =====
            function formatAIResponse(text) {
                if (!text) return '';
                
                let html = '';
                let remainingText = text;
                
                // Try to extract and format the first JSON object (product info)
                const jsonObjMatch = remainingText.match(/^\s*\{[\s\S]*?\}\s*/);
                if (jsonObjMatch) {
                    try {
                        const jsonStr = jsonObjMatch[0];
                        const jsonObj = JSON.parse(jsonStr);
                        
                        // Create a nice table for the JSON object
                        html += '<table class="info-table">';
                        for (const [key, value] of Object.entries(jsonObj)) {
                            const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            html += '<tr><th>' + escapeHtml(formattedKey) + '</th><td>' + escapeHtml(String(value)) + '</td></tr>';
                        }
                        html += '</table>';
                        
                        remainingText = remainingText.substring(jsonObjMatch[0].length);
                    } catch (e) {
                        // JSON parse failed, skip
                    }
                }
                
                // Try to extract description paragraph (text before the credentials array)
                const descMatch = remainingText.match(/^([\s\S]*?)(?=\[\s*\{|\[\s*\]|Product identification:|$)/);
                if (descMatch && descMatch[1].trim()) {
                    const descText = descMatch[1].trim();
                    if (descText) {
                        html += '<hr class="section-divider">';
                        html += '<div class="section-content">' + formatTextContent(descText) + '</div>';
                        remainingText = remainingText.substring(descMatch[0].length);
                    }
                }
                
                // Try to extract credentials array
                const credsArrayMatch = remainingText.match(/\[\s*\{[\s\S]*?\}\s*\]/);
                if (credsArrayMatch) {
                    try {
                        const credsArray = JSON.parse(credsArrayMatch[0]);
                        if (Array.isArray(credsArray) && credsArray.length > 0 && credsArray[0].username !== undefined) {
                            html += '<hr class="section-divider">';
                            html += '<div class="section-title">Default Credentials</div>';
                            html += '<table class="creds-table">';
                            html += '<tr><th>Scope</th><th>Username</th><th>Password</th><th>Notes</th></tr>';
                            for (const cred of credsArray) {
                                html += '<tr>';
                                html += '<td>' + escapeHtml(cred.scope || '') + '</td>';
                                html += '<td>' + escapeHtml(cred.username || '') + '</td>';
                                html += '<td>' + escapeHtml(cred.password || '') + '</td>';
                                html += '<td class="notes">' + escapeHtml(cred.notes || '') + '</td>';
                                html += '</tr>';
                            }
                            html += '</table>';
                        }
                        remainingText = remainingText.substring(remainingText.indexOf(credsArrayMatch[0]) + credsArrayMatch[0].length);
                    } catch (e) {
                        // JSON parse failed, skip
                    }
                }
                
                // Remove empty arrays []
                remainingText = remainingText.replace(/\[\s*\]/g, '');
                
                // Process remaining sections (Product identification:, What this product is used for:, etc.)
                const sectionPattern = /([A-Z][^:]+):\s*([\s\S]*?)(?=(?:[A-Z][^:]+:|$))/g;
                let sectionMatch;
                let sectionsHtml = '';
                
                while ((sectionMatch = sectionPattern.exec(remainingText)) !== null) {
                    const sectionTitle = sectionMatch[1].trim();
                    const sectionContent = sectionMatch[2].trim();
                    
                    if (sectionContent) {
                        sectionsHtml += '<hr class="section-divider">';
                        sectionsHtml += '<div class="section-title">' + escapeHtml(sectionTitle) + '</div>';
                        sectionsHtml += '<div class="section-content">' + formatTextContent(sectionContent) + '</div>';
                    }
                }
                
                if (sectionsHtml) {
                    html += sectionsHtml;
                }
                
                // If no formatting was applied, just return escaped text
                if (!html) {
                    html = '<div class="section-content">' + formatTextContent(text) + '</div>';
                }
                
                return html;
            }
            
            // ===== Helper: Escape HTML =====
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            // ===== Helper: Format text content with basic styling =====
            function formatTextContent(text) {
                let html = escapeHtml(text);
                
                // Bold text **text**
                html = html.replace(/\*\*([^*]+)\*\*/g, '<span class="highlight">$1</span>');
                
                // Credential format: word / word
                html = html.replace(/(\w+)\s*\/\s*(\w+)(?=\s*\(|\s*-|\s*,|\s*$)/g, '<span class="credential">$1 / $2</span>');
                
                // Convert line breaks
                html = html.replace(/\n/g, '<br>');
                
                // Convert bullet points
                html = html.replace(/^- /gm, '• ');
                
                return html;
            }
            
            function closeAIResponseModal() {
                const modal = document.getElementById('aiResponseModal');
                if (modal) {
                    modal.classList.remove('active');
                }
            }
        </script>
    </body>
    </html>
    """
    
    # ===== Flask route: main page =====
    @app.route('/')
    def index():
        """Render main page with all groups."""
        return render_template_string(
            HTML_TEMPLATE, 
            groups=web_groups,
            total_images=total_images,
            multi_image_groups=multi_image_groups,
            ai_responses=ai_responses,
            custom_prompt=custom_prompt
        )
    
    # ===== Flask route: serve full-size images =====
    @app.route('/image/<filename>')
    def serve_image(filename):
        """Serve original full-size image file."""
        if filename in image_paths:
            return send_file(image_paths[filename])
        return "Image not found", 404
    
    # ===== Flask route: save group name =====
    @app.route('/rename_group', methods=['POST'])
    def rename_group():
        """Save user-defined group name."""
        try:
            data = request.json
            seed = data['seed']
            name = data['name']
            
            # Update in-memory and save to disk
            group_names[seed] = name
            save_group_names(group_names, groups_path)
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # ===== Flask route: save API key =====
    @app.route('/save_api_key', methods=['POST'])
    def save_api_key():
        """Save API key for selected AI provider."""
        try:
            data = request.json
            provider = data.get('provider')
            api_key = data.get('api_key')
            
            if not provider or not api_key:
                return jsonify({'success': False, 'error': 'Missing provider or API key'})
            
            # Load existing API keys
            api_keys_path = os.path.join(folder_path, '.api_keys.json')
            api_keys = {}
            if os.path.exists(api_keys_path):
                try:
                    with open(api_keys_path, 'r') as f:
                        api_keys = json.load(f)
                except:
                    pass
            
            # Save new key
            api_keys[provider] = api_key
            
            with open(api_keys_path, 'w') as f:
                json.dump(api_keys, f, indent=2)
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # ===== Flask route: get API key =====
    @app.route('/get_api_key', methods=['POST'])
    def get_api_key():
        """Get API key for selected AI provider."""
        try:
            data = request.json
            provider = data.get('provider')
            
            if not provider:
                return jsonify({'success': False, 'error': 'Missing provider'})
            
            # Load API keys
            api_keys_path = os.path.join(folder_path, '.api_keys.json')
            if os.path.exists(api_keys_path):
                try:
                    with open(api_keys_path, 'r') as f:
                        api_keys = json.load(f)
                        api_key = api_keys.get(provider)
                        if api_key:
                            return jsonify({'success': True, 'api_key': api_key})
                except:
                    pass
            
            return jsonify({'success': True, 'api_key': None})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # ===== Flask route: test AI connection =====
    @app.route('/test_ai_connection', methods=['POST'])
    def test_ai_connection():
        """Test connection to selected AI provider using API key."""
        try:
            data = request.json
            provider = data.get('provider', 'perplexity')
            model = data.get('model', 'sonar')
            
            # Load API key
            api_keys_path = os.path.join(folder_path, '.api_keys.json')
            api_key = None
            if os.path.exists(api_keys_path):
                try:
                    with open(api_keys_path, 'r') as f:
                        api_keys = json.load(f)
                        api_key = api_keys.get(provider)
                except:
                    pass
            
            if not api_key:
                return jsonify({
                    'success': False,
                    'error': 'No API key found. Please save your API key first.'
                })
            
            # Test connection based on provider
            if provider == 'perplexity':
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Simple test request to Perplexity API
                test_url = 'https://api.perplexity.ai/chat/completions'
                test_payload = {
                    'model': model,
                    'messages': [
                        {'role': 'system', 'content': 'Be brief.'},
                        {'role': 'user', 'content': 'Hi'}
                    ]
                }
                
                try:
                    response = requests.post(test_url, headers=headers, json=test_payload, timeout=10)
                    
                    if response.status_code == 200:
                        return jsonify({
                            'success': True,
                            'provider': provider,
                            'model': model
                        })
                    else:
                        error_detail = response.text
                        return jsonify({
                            'success': False,
                            'error': f'API error: {response.status_code} - {error_detail}'
                        })
                        
                except requests.exceptions.RequestException as e:
                    return jsonify({
                        'success': False,
                        'error': f'Connection error: {str(e)}'
                    })
            
            elif provider == 'claude':
                headers = {
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json'
                }
                
                # Test request to Claude API
                test_url = 'https://api.anthropic.com/v1/messages'
                test_payload = {
                    'model': model,
                    'max_tokens': 1,
                    'messages': [
                        {'role': 'user', 'content': 'Hello'}
                    ]
                }
                
                try:
                    response = requests.post(test_url, headers=headers, json=test_payload, timeout=10)
                    
                    if response.status_code == 200:
                        return jsonify({
                            'success': True,
                            'provider': provider,
                            'model': model
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'API error: {response.status_code}'
                        })
                        
                except requests.exceptions.RequestException as e:
                    return jsonify({
                        'success': False,
                        'error': f'Connection error: {str(e)}'
                    })
            
            elif provider == 'chatgpt':
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Test request to OpenAI API
                test_url = 'https://api.openai.com/v1/chat/completions'
                test_payload = {
                    'model': model,
                    'messages': [
                        {'role': 'user', 'content': 'Hello'}
                    ],
                    'max_tokens': 1
                }
                
                try:
                    response = requests.post(test_url, headers=headers, json=test_payload, timeout=10)
                    
                    if response.status_code == 200:
                        return jsonify({
                            'success': True,
                            'provider': provider,
                            'model': model
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'API error: {response.status_code}'
                        })
                        
                except requests.exceptions.RequestException as e:
                    return jsonify({
                        'success': False,
                        'error': f'Connection error: {str(e)}'
                    })
            
            return jsonify({
                'success': False,
                'error': 'Unknown provider'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # ===== Flask route: save custom prompt =====
    @app.route('/save_custom_prompt', methods=['POST'])
    def save_custom_prompt_route():
        """Save custom AI prompt to disk."""
        try:
            data = request.json
            prompt = data.get('prompt', '')
            
            save_custom_prompt(prompt, custom_prompt_path)
            
            # Update the in-memory custom_prompt variable
            nonlocal custom_prompt
            custom_prompt = prompt
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # ===== Flask route: search with AI =====
    @app.route('/search_with_ai', methods=['POST'])
    def search_with_ai():
        """Send images to selected AI provider for analysis."""
        try:
            data = request.json
            provider = data.get('provider', 'perplexity')
            model = data.get('model', 'sonar')
            group_name = data.get('group_name', 'Unknown Group')
            seed_filename = data.get('seed_filename', '')
            images = data.get('images', [])
            
            if not images:
                return jsonify({
                    'success': False,
                    'error': 'No images provided'
                })
            
            # Load API key
            api_keys_path = os.path.join(folder_path, '.api_keys.json')
            api_key = None
            if os.path.exists(api_keys_path):
                try:
                    with open(api_keys_path, 'r') as f:
                        api_keys = json.load(f)
                        api_key = api_keys.get(provider)
                except:
                    pass
            
            if not api_key:
                return jsonify({
                    'success': False,
                    'error': 'No API key found. Please configure your API key in the Advanced menu.'
                })
            
            # Build the prompt - use custom prompt if defined, otherwise use default
            if custom_prompt and custom_prompt.strip():
                # Use custom prompt as the user prompt
                user_prompt = custom_prompt.strip()
                system_prompt = "You are an AI assistant analyzing a screenshot. Be concise but informative."
            else:
                # Default prompts
                system_prompt = """You are an AI assistant analyzing a screenshot. 
Analyze the image and provide insights about:
1. What application or website is shown
2. What the screenshot is depicting (UI elements, functionality, content)
3. Any notable patterns or issues you observe

Be concise but informative."""
                user_prompt = f"Analyze this screenshot from group '{group_name}'. Describe what you see and any notable observations."
            
            ai_response = None
            
            # Call the appropriate AI provider
            if provider == 'perplexity':
                # Perplexity supports images via URL or base64
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Build content with images
                content = []
                for img in images[:5]:  # Limit to 5 images to avoid token limits
                    content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:{img['media_type']};base64,{img['data']}"
                        }
                    })
                content.append({
                    'type': 'text',
                    'text': user_prompt
                })
                
                payload = {
                    'model': model,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': content}
                    ],
                    'max_tokens': 1000
                }
                
                try:
                    response = requests.post(
                        'https://api.perplexity.ai/chat/completions',
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        ai_response = result.get('choices', [{}])[0].get('message', {}).get('content', 'No response received')
                        
                        # Save response to cache
                        if seed_filename:
                            ai_responses[seed_filename] = ai_response
                            save_ai_responses(ai_responses, ai_responses_path)
                        
                        return jsonify({
                            'success': True,
                            'response': ai_response
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'Perplexity API error: {response.status_code} - {response.text}'
                        })
                        
                except requests.exceptions.RequestException as e:
                    return jsonify({
                        'success': False,
                        'error': f'Connection error: {str(e)}'
                    })
            
            elif provider == 'claude':
                headers = {
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json'
                }
                
                # Build content with images for Claude
                content = []
                for img in images[:5]:  # Limit to 5 images
                    content.append({
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': img['media_type'],
                            'data': img['data']
                        }
                    })
                content.append({
                    'type': 'text',
                    'text': user_prompt
                })
                
                payload = {
                    'model': model,
                    'max_tokens': 1000,
                    'system': system_prompt,
                    'messages': [
                        {'role': 'user', 'content': content}
                    ]
                }
                
                try:
                    response = requests.post(
                        'https://api.anthropic.com/v1/messages',
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        ai_response = result.get('content', [{}])[0].get('text', 'No response received')
                        
                        # Save response to cache
                        if seed_filename:
                            ai_responses[seed_filename] = ai_response
                            save_ai_responses(ai_responses, ai_responses_path)
                        
                        return jsonify({
                            'success': True,
                            'response': ai_response
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'Claude API error: {response.status_code} - {response.text}'
                        })
                        
                except requests.exceptions.RequestException as e:
                    return jsonify({
                        'success': False,
                        'error': f'Connection error: {str(e)}'
                    })
            
            elif provider == 'chatgpt':
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Build content with images for OpenAI
                content = []
                for img in images[:5]:  # Limit to 5 images
                    content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:{img['media_type']};base64,{img['data']}",
                            'detail': 'low'  # Use low detail to reduce tokens
                        }
                    })
                content.append({
                    'type': 'text',
                    'text': user_prompt
                })
                
                payload = {
                    'model': model,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': content}
                    ],
                    'max_tokens': 1000
                }
                
                try:
                    response = requests.post(
                        'https://api.openai.com/v1/chat/completions',
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        ai_response = result.get('choices', [{}])[0].get('message', {}).get('content', 'No response received')
                        
                        # Save response to cache
                        if seed_filename:
                            ai_responses[seed_filename] = ai_response
                            save_ai_responses(ai_responses, ai_responses_path)
                        
                        return jsonify({
                            'success': True,
                            'response': ai_response
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'OpenAI API error: {response.status_code} - {response.text}'
                        })
                        
                except requests.exceptions.RequestException as e:
                    return jsonify({
                        'success': False,
                        'error': f'Connection error: {str(e)}'
                    })
            
            return jsonify({
                'success': False,
                'error': 'Unknown provider'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # ===== Start Flask server =====
    print(f"Starting web server at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=port, debug=False)

# ===== Main entry point =====
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <folder_path> [exact_threshold] [structural_threshold] [port]")
        print("  folder_path: Directory containing screenshots")
        print("  exact_threshold: Exact duplicate threshold (default: 97)")
        print("  structural_threshold: Structural similarity threshold (default: 97)")
        print("  port: Web server port (default: 5000)")
        sys.exit(1)
    
    folder = sys.argv[1]
    exact_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 97
    structural_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 97
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    
    # Validate folder path
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory")
        sys.exit(1)
    
    # Run similarity analysis
    groups = group_similar_images(folder, exact_threshold, structural_threshold)
    
    # Launch web interface if groups found
    if groups:
        create_web_interface(groups, folder, port)
    else:
        print("No images to display")
