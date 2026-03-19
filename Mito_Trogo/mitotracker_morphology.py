"""Microscopy processing for mitotracker + multicell segmentation.

This script loads all TIFF files in a folder and performs the following steps:

1) Finds the largest cell mask using channel 4 (assumes one large object).
   - Keeps small waves/disturbances around the boundary using a mild closing.
2) Computes total orange signal (channel 2) inside the cell mask.
3) Finds punctate green signal (channel 3) while ignoring very large green blobs.
4) Segments mitochondria (channel 1) and computes basic morphology metrics.
5) Generates visualization figures so segmentation decisions can be inspected.

Usage:
	python mitotracker_morphology.py --input-folder ./Mito_Trogo --outdir ./results

Dependencies:
	numpy, scipy, scikit-image, matplotlib, tifffile
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation
from skimage.filters import frangi, threshold_local, threshold_sauvola, threshold_niblack
from skimage.feature import blob_log
from skimage.draw import disk


def _read_tif(path: Path) -> np.ndarray:
	"""Read a TIFF and return an ndarray.

	Handles multi-page and multi-channel TIFFs.
	"""

	# tifffile automatically handles multi-page / multi-channel!
	arr = tifffile.imread(path)

	if arr.ndim == 2:
		# Single-plane intensity image
		return arr

	# If we have a stack (Z, Y, X, C) or (C, Y, X) or (Y, X, C), normalize to (C, Y, X)
	if arr.ndim == 3:
		# Either (C, Y, X) or (Y, X, C)
		if arr.shape[0] <= 4:
			return arr
		# Otherwise assume last axis is channels
		return np.transpose(arr, (2, 0, 1))

	if arr.ndim == 4:
		# Common: (Z, Y, X, C) or (C, Z, Y, X)
		if arr.shape[-1] <= 4:
			# collapse Z and use channels last
			arr = np.squeeze(arr)
			if arr.ndim == 3 and arr.shape[-1] <= 4:
				return np.transpose(arr, (2, 0, 1))
		# Fall back to taking the first z-plane
		arr = arr[0]
		if arr.ndim == 3 and arr.shape[-1] <= 4:
			return np.transpose(arr, (2, 0, 1))
		return arr

	raise ValueError(f"Unsupported image shape: {arr.shape}")


def find_tifs(folder: Path) -> List[Path]:
	# Only include real TIFFs; skip pointer files (e.g. Git LFS pointers) or broken files.
	paths = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
	valid = []
	for p in paths:
		try:
			with open(p, "rb") as f:
				head = f.read(4)
			# TIFF files start with II (little endian) or MM (big endian)
			if head[:2] in (b"II", b"MM"):
				valid.append(p)
			else:
				# Skip LFS pointer / text files
				print(f"Skipping non-TIFF file: {p} (starts {head!r})")
		except Exception as e:
			print(f"Could not read {p}: {e}")
	return valid


def segment_cell(
	ch4: np.ndarray,
	threshold: Optional[float] = None,
	threshold_method: str = "otsu",
	threshold_value: Optional[float] = None,
	threshold_percentile: float = 99.0,
	threshold_local_block: int = 51,
	threshold_local_offset: float = 0.0,
	gaussian_sigma: float = 0.0,
	closing_radius: int = 10,
	fill_holes: bool = True,
	min_area: int = 50_000,
	keep_largest: bool = True,
	max_area: Optional[int] = None,
	verbose: bool = False,
) -> np.ndarray:
	"""Segment the largest cell from channel 4.

	The goal is to segment a single large object and keep small waves on the boundary.
	"""

	proc = ch4.astype(float)

	if gaussian_sigma > 0:
		proc = ndi.gaussian_filter(proc, sigma=gaussian_sigma)

	def _make_mask(thresh: float) -> np.ndarray:
		mask = proc > thresh
		if closing_radius > 0:
			mask = morphology.binary_closing(mask, footprint=morphology.disk(closing_radius))
		if fill_holes:
			mask = ndi.binary_fill_holes(mask)
		return mask

	def _extract_largest(mask: np.ndarray, keep_largest: bool = True) -> Tuple[np.ndarray, int]:
		label = measure.label(mask)
		if label.max() == 0:
			return mask.astype(bool), 0
		props = sorted(measure.regionprops(label), key=lambda p: p.area, reverse=True)
		if not props:
			return mask.astype(bool), 0
		largest = props[0]
		if keep_largest:
			mask_out = label == largest.label
		else:
			# Keep all objects except the largest one, then take the new largest
			other_labels = [p.label for p in props[1:]]
			if other_labels:
				other_mask = np.isin(label, other_labels)
				other_props = sorted(measure.regionprops(measure.label(other_mask)), key=lambda p: p.area, reverse=True)
				if other_props:
					largest = other_props[0]
					mask_out = label == largest.label
				else:
					mask_out = mask.astype(bool)
			else:
				mask_out = mask.astype(bool)
		mask_out = morphology.remove_small_objects(mask_out, min_size=min_area)
		return mask_out, int(largest.area)

	def _attempt(method: str, override_threshold: Optional[float] = None, **kwargs) -> Tuple[np.ndarray, int]:
		"""Try one thresholding method and return (mask, largest_component_area)."""
		try:
			if override_threshold is not None:
				threshold_value = override_threshold
			elif method == "otsu":
				threshold_value = filters.threshold_otsu(proc)
			elif method == "percentile":
				threshold_value = np.percentile(proc, kwargs.get("percentile", threshold_percentile))
			elif method == "local":
				threshold_value = threshold_local(
					proc,
					block_size=kwargs.get("block_size", threshold_local_block),
					method="gaussian",
					offset=kwargs.get("offset", threshold_local_offset),
				)
			else:
				raise ValueError(f"Unknown threshold method: {method}")

			mask = _make_mask(threshold_value)
			return _extract_largest(mask, keep_largest=params.get("keep_largest", keep_largest))
		except Exception as e:
			if verbose:
				print(f"Cell segmentation attempt failed ({method}): {e}")
			return np.zeros_like(proc, dtype=bool), 0

	# Build attempt list: fixed threshold (if specified) + requested method + fallbacks
	attempts: List[Tuple[str, dict]] = []
	if threshold_value is not None:
		attempts.append(("fixed", {"override_threshold": threshold_value}))
	attempts.append((threshold_method, {}))
	for fallback in ["otsu", "percentile", "local"]:
		if fallback != threshold_method:
			attempts.append((fallback, {}))

	# Add some common parameter variants for percentile/local to improve robustness
	for pct in [threshold_percentile, 99.5, 98.0, 95.0, 90.0, 85.0, 80.0, 75.0]:
		if pct != threshold_percentile:
			attempts.append(("percentile", {"percentile": pct}))
	for offset in [threshold_local_offset, -0.05, 0.0, 0.05, -0.1, -0.2, -0.3]:
		if offset != threshold_local_offset:
			attempts.append(("local", {"block_size": threshold_local_block, "offset": offset}))
	# Also try with keep_largest=False for some, to allow smaller cells if largest is too big
	for pct in [95.0, 90.0, 85.0]:
		attempts.append(("percentile", {"percentile": pct, "keep_largest": False}))
	for offset in [-0.05, -0.1, -0.2]:
		attempts.append(("local", {"block_size": threshold_local_block, "offset": offset, "keep_largest": False}))

	# Try all attempts until we get a large enough cell
	# But avoid selecting the entire image as the cell
	image_area = proc.size
	max_cell_area = max_area if max_area is not None else int(0.5 * image_area)  # Default to 50% if not specified
	final_mask = np.zeros_like(proc, dtype=bool)
	for method, params in attempts:
		mask, largest_area = _attempt(method, **params)
		if verbose:
			print(f"Cell segmentation attempt: method={method}, largest_area={largest_area}")
		if min_area <= largest_area <= max_cell_area:
			return mask
		final_mask = mask

	# Fallback: return the last computed mask even if it was small or large (helps debugging)
	return final_mask


def compute_orange_signal(ch2: np.ndarray, cell_mask: np.ndarray) -> Dict[str, float]:
	values = ch2[cell_mask]
	return {
		"total_intensity": float(values.sum()),
		"mean_intensity": float(values.mean()) if values.size else 0.0,
		"pixel_count": int(values.size),
	}


def segment_green_puncta(
	ch3: np.ndarray,
	threshold: Optional[float] = None,
	min_size: int = 5,
	max_size: int = 5_000,
	cell_mask: Optional[np.ndarray] = None,
	threshold_method: str = "otsu",
	threshold_percentile: float = 99.5,
	gaussian_sigma: float = 1.0,
	local_bg_sigma: float = 10.0,
	fixed_threshold: Optional[float] = 5.0,
) -> Tuple[np.ndarray, measure._regionprops.RegionProperties]:
	"""Segment punctate green spots while ignoring extremely large green blobs."""

	# Optionally restrict to within the cell to reduce background noise
	if cell_mask is not None:
		ch3 = np.where(cell_mask, ch3, 0)

	# Smooth a bit to reduce speckle
	if gaussian_sigma > 0:
		ch3 = ndi.gaussian_filter(ch3.astype(float), sigma=gaussian_sigma)

	# Subtract local background to make threshold relative
	if local_bg_sigma > 0:
		background = ndi.gaussian_filter(ch3, sigma=local_bg_sigma)
		ch3 = ch3 - background
		ch3 = np.clip(ch3, 0, None)  # Ensure non-negative

	# Use fixed threshold if provided, else adaptive
	if fixed_threshold is not None:
		threshold = fixed_threshold
	elif threshold is None:
		if threshold_method == "otsu":
			threshold = filters.threshold_otsu(ch3)
		elif threshold_method == "percentile":
			threshold = np.percentile(ch3, threshold_percentile)
		else:
			raise ValueError(f"Unknown threshold_method: {threshold_method}")
	mask = ch3 > threshold

	label = measure.label(mask)
	props = measure.regionprops(label, intensity_image=ch3)

	keep = np.zeros_like(mask, dtype=bool)
	kept_props = []
	for p in props:
		if min_size <= p.area <= max_size:
			keep[label == p.label] = True
			kept_props.append(p)

	return keep, kept_props



def _skeleton_segments(mask: np.ndarray, prune_junctions: bool = True) -> Tuple[np.ndarray, np.ndarray]:
	"""Return connected skeleton segments derived from a binary mask.

	When `prune_junctions` is True, pixels where the skeleton branches are removed
	so that each segment is an individual filament between branch points.
	"""

	skel = morphology.skeletonize(mask)
	if not prune_junctions:
		return measure.label(skel, connectivity=2), skel

	# Count 8-connected neighbors for each skeleton pixel
	neighbor_count = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3)), mode="constant") - skel
	branch_points = (skel & (neighbor_count > 2))
	skel_no_branch = skel.copy()
	skel_no_branch[branch_points] = 0
	labels = measure.label(skel_no_branch, connectivity=2)
	return labels, skel


def prune_skeleton_by_length(skel: np.ndarray, min_length: int = 10) -> np.ndarray:
	"""Remove skeleton segments shorter than `min_length` pixels."""

	labels = measure.label(skel, connectivity=2)
	props = measure.regionprops(labels)
	pruned = skel.copy()
	for p in props:
		if p.area < min_length:
			pruned[labels == p.label] = 0
	return pruned



def segment_mitochondria(
	ch1: np.ndarray,
	threshold: Optional[float] = None,
	min_size: int = 5,
	closing_radius: int = 1,
	enhance: Optional[str] = "none",
	threshold_method: str = "percentile",
	threshold_percentile: float = 98.0,
	threshold_local_block: int = 51,
	threshold_local_offset: float = 0.0,
	threshold_sauvola_window: int = 25,
	threshold_sauvola_k: float = 0.2,
	threshold_niblack_window: int = 25,
	threshold_niblack_k: float = 0.2,
	contrast_sigma: float = 10.0,
	gaussian_sigma: float = 0.5,
	prune_skeleton: bool = True,
	skeleton_min_length: int = 5,
	min_mean_intensity: float = 0.2,
	verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Segment mitochondrial objects in the red channel.

	Returns (mito_mask, mito_skeleton).
	"""
	# Normalize the input so that thresholds behave consistently across images
	proc = ch1.astype(float)
	if proc.max() > 0:
		proc = proc / proc.max()

	# Smooth to reduce noise
	if gaussian_sigma > 0:
		proc = ndi.gaussian_filter(proc, sigma=gaussian_sigma)

	# Enhance local contrast (helps pick up dim filaments vs background)
	if contrast_sigma > 0:
		background = ndi.gaussian_filter(proc, sigma=contrast_sigma)
		proc = proc - background
		proc = np.clip(proc, 0, 1)

	# -------- filamentous component (Frangi) --------
	filaments = proc
	if enhance == "frangi":
		filaments = frangi(proc)

	# Threshold the filament enhancement
	if threshold is not None:
		fil_thresh = threshold
	elif threshold_method == "otsu":
		fil_thresh = filters.threshold_otsu(filaments)
	elif threshold_method == "percentile":
		fil_thresh = np.percentile(filaments, threshold_percentile)
	else:
		raise ValueError(f"Unknown threshold_method: {threshold_method}")

	mask_filaments = filaments > fil_thresh

	# -------- punctate component (LoG blobs) --------
	# Use the normalized smoothed image to detect small bright blobs.
	blobs = blob_log(proc, min_sigma=1, max_sigma=5, num_sigma=4, threshold=0.05)
	mask_blobs = np.zeros_like(proc, dtype=bool)
	for y, x, sigma in blobs:
		r = int(np.ceil(np.sqrt(2) * sigma))
		rr, cc = disk((int(y), int(x)), r, shape=proc.shape)
		mask_blobs[rr, cc] = True

	# Combine filament and puncta masks
	mask = mask_filaments | mask_blobs

	# Clean up the mask
	if closing_radius > 0:
		mask = morphology.binary_closing(mask, footprint=morphology.disk(closing_radius))
	mask = morphology.remove_small_objects(mask, min_size=min_size)

	# Remove objects with very low mean intensity (likely background noise)
	if min_mean_intensity > 0:
		labels = measure.label(mask)
		props = measure.regionprops(labels, intensity_image=proc)
		for p in props:
			if p.mean_intensity < min_mean_intensity:
				mask[labels == p.label] = False

	# Skeletonize for morphology metrics
	skel = morphology.skeletonize(mask)
	if prune_skeleton and skeleton_min_length > 0:
		skel = prune_skeleton_by_length(skel, min_length=skeleton_min_length)
	# Ensure skeleton stays within the mask (though it should by default)
	skel = skel & mask

	if verbose:
		print(f"mito seg: fil_thresh={fil_thresh:.3g} min_size={min_size} prune={prune_skeleton} mean_intensity_cutoff={min_mean_intensity}")

	return mask, skel


def compute_lacunarity(mask: np.ndarray, box_sizes: List[int] = [4, 8, 16, 32]) -> Dict[int, float]:
	"""Compute lacunarity for a binary mask using gliding boxes.

	This is a simple implementation. Lacunarity is high when the object distribution
	is heterogeneous (large gaps).
	"""

	lac = {}
	mask = mask.astype(np.uint8)
	for box in box_sizes:
		if box > min(mask.shape):
			continue
		# Use uniform filter on the binary mask
		summed = ndi.uniform_filter(mask.astype(float), size=box, mode="constant") * (box * box)
		mean = np.mean(summed)
		var = np.var(summed)
		lac[box] = float(var / (mean * mean + 1e-12) + 1)
	return lac



def compute_mito_metrics(mito_mask: np.ndarray, mito_skel: np.ndarray, cell_mask: np.ndarray) -> Dict[str, Any]:
	"""Compute mitochondrial morphology metrics using the mask + skeleton."""

	# Basic mask-based stats
	label = measure.label(mito_mask)
	props = measure.regionprops(label)

	cell_props = measure.regionprops(measure.label(cell_mask))
	if cell_props:
		cell_centroid = np.array(cell_props[0].centroid)
	else:
		cell_centroid = np.array(mito_mask.shape) / 2

	centroids = np.array([p.centroid for p in props])
	distances = np.linalg.norm(centroids - cell_centroid, axis=1) if centroids.size else np.array([])

	# Skeleton based metrics
	# Break skeleton into segments by connected components
	skel_labels = measure.label(mito_skel, connectivity=2)
	skel_props = measure.regionprops(skel_labels)
	skel_lengths = [p.area for p in skel_props]

	metrics = {
		"num_mito_objects": len(props),
		"mean_area": float(np.mean([p.area for p in props])) if props else 0.0,
		"median_area": float(np.median([p.area for p in props])) if props else 0.0,
		"num_skel_segments": len(skel_props),
		"mean_skel_length": float(np.mean(skel_lengths)) if skel_lengths else 0.0,
		"median_skel_length": float(np.median(skel_lengths)) if skel_lengths else 0.0,
		"total_skel_length": float(np.sum(skel_lengths)),
		"mean_distance_to_cell_center": float(distances.mean()) if distances.size else 0.0,
		"lacunarity": compute_lacunarity(mito_mask),
	}
	return metrics


def overlay_mask(ax, mask: np.ndarray, color: str = "red", alpha: float = 0.25):
	ax.imshow(mask, cmap="gray", alpha=0)
	ax.contour(mask, colors=[color], linewidths=1)


def visualize(
	ch1: np.ndarray,
	ch2: np.ndarray,
	ch3: np.ndarray,
	ch4: np.ndarray,
	cell_mask: np.ndarray,
	green_mask: np.ndarray,
	mito_mask: np.ndarray,
	mito_skel: np.ndarray,
	out_path: Optional[Path] = None,
):
	fig, axes = plt.subplots(2, 3, figsize=(15, 10))
	axes = axes.ravel()

	axes[0].imshow(ch4, cmap="gray")
	axes[0].set_title("Channel 4 (cell seg)")
	overlay_mask(axes[0], cell_mask, color="cyan")

	axes[1].imshow(ch2, cmap="magma")
	axes[1].set_title("Channel 2 (orange)")
	overlay_mask(axes[1], cell_mask, color="cyan")

	axes[2].imshow(ch3, cmap="Greens", vmin=0, vmax=np.percentile(ch3, 99.5))
	axes[2].set_title("Channel 3 (green)")
	overlay_mask(axes[2], green_mask, color="yellow")

	axes[3].imshow(ch1, cmap="Reds")
	axes[3].set_title("Channel 1 (mito)")
	overlay_mask(axes[3], mito_mask, color="lime")
	# overlay skeleton for filament resolution (use a fixed color to avoid colormap artifacts)
	axes[3].contour(mito_skel, levels=[0.5], colors=["magenta"], linewidths=0.5)

	axes[4].imshow(ch1, cmap="Reds")
	axes[4].set_title("Mito + Cell boundary")
	overlay_mask(axes[4], cell_mask, color="cyan")
	overlay_mask(axes[4], mito_mask, color="lime")
	axes[4].contour(mito_skel, levels=[0.5], colors=["magenta"], linewidths=0.5)

	axes[5].axis("off")
	axes[5].text(0.05, 0.5, "Segmentation overlays:\n- cyan: cell\n- yellow: green puncta\n- lime: mito", fontsize=12)

	for ax in axes:
		ax.axis("off")

	plt.tight_layout()
	if out_path is not None:
		fig.savefig(out_path, dpi=150)
	plt.close(fig)


def create_channel_montages(ch1s: List[np.ndarray], ch2s: List[np.ndarray], ch3s: List[np.ndarray], ch4s: List[np.ndarray], cell_masks: List[np.ndarray], green_masks: List[np.ndarray], outdir: Path):
	"""Create montage images for each channel across all files."""
	import math

	def make_montage(images: List[np.ndarray], title: str, filename: str, cmap='gray'):
		if not images:
			return
		n = len(images)
		cols = math.ceil(math.sqrt(n))
		rows = math.ceil(n / cols)
		fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
		if n == 1:
			axes = [axes]
		else:
			axes = axes.flatten()
		for i, img in enumerate(images):
			axes[i].imshow(img, cmap=cmap)
			axes[i].axis('off')
		for i in range(n, len(axes)):
			axes[i].axis('off')
		plt.suptitle(title)
		plt.tight_layout()
		fig.savefig(outdir / filename, dpi=150)
		plt.close(fig)

	make_montage(ch4s, "Channel 4 (Cell)", "channel4_montage.png")
	make_montage(ch2s, "Channel 2 (Orange)", "channel2_montage.png")
	make_montage(ch3s, "Channel 3 (Green)", "channel3_montage.png")
	make_montage(ch1s, "Channel 1 (Mito)", "channel1_montage.png")
	make_montage(cell_masks, "Cell Masks", "cell_masks_montage.png")
	make_montage(green_masks, "Green Puncta Masks", "green_masks_montage.png", cmap='plasma')


def process_file(
	path: Path,
	outdir: Path,
	cell_min_area: int = 50_000,
	cell_max_area: Optional[int] = None,
	cell_closing: int = 10,
	cell_threshold_method: str = "otsu",
	cell_threshold_value: Optional[float] = None,
	cell_threshold_percentile: float = 99.0,
	cell_threshold_local_block: int = 51,
	cell_threshold_local_offset: float = 0.0,
	cell_gaussian_sigma: float = 0.0,
	cell_keep_largest: bool = True,
	cell_verbose: bool = False,
	green_min_size: int = 5,
	green_max_size: int = 5_000,
	mito_min_size: int = 50,
	green_threshold: Optional[float] = None,
	green_threshold_method: str = "percentile",
	green_threshold_percentile: float = 99.5,
	green_gaussian_sigma: float = 1.0,
	green_local_bg_sigma: float = 10.0,
	green_fixed_threshold: Optional[float] = None,
	mito_enhance: str = "none",
	mito_threshold_method: str = "percentile",
	mito_threshold_percentile: float = 90.0,
	mito_threshold_local_block: int = 51,
	mito_threshold_local_offset: float = 0.0,
	mito_threshold_sauvola_window: int = 25,
	mito_threshold_sauvola_k: float = 0.2,
	mito_threshold_niblack_window: int = 25,
	mito_threshold_niblack_k: float = 0.2,
	mito_contrast_sigma: float = 10.0,
	mito_gaussian_sigma: float = 0.0,
	mito_prune_skeleton: bool = False,
	mito_skeleton_min_length: int = 10,
	mito_min_mean_intensity: float = 0.0,
	mito_verbose: bool = False,
):
	img = _read_tif(path)
	if img.ndim != 3 or img.shape[0] < 4:
		raise ValueError(f"Expected 4-channel image in {path}, got shape {img.shape}")

	ch1, ch2, ch3, ch4 = img[0], img[1], img[2], img[3]

	cell_mask = segment_cell(
		ch4,
		min_area=cell_min_area,
		max_area=cell_max_area,
		closing_radius=cell_closing,
		threshold_method=cell_threshold_method,
		threshold_value=cell_threshold_value,
		threshold_percentile=cell_threshold_percentile,
		threshold_local_block=cell_threshold_local_block,
		threshold_local_offset=cell_threshold_local_offset,
		gaussian_sigma=cell_gaussian_sigma,
		keep_largest=cell_keep_largest,
		verbose=cell_verbose,
	)
	orange_stats = compute_orange_signal(ch2, cell_mask)

	# Restrict all downstream analysis to the detected cell region.
	# This prevents background outside the cell from affecting segmentation and metrics.
	ch1_in = np.where(cell_mask, ch1, 0)
	ch2_in = np.where(cell_mask, ch2, 0)
	ch3_in = np.where(cell_mask, ch3, 0)

	green_mask, green_props = segment_green_puncta(
		ch3_in,
		min_size=green_min_size,
		max_size=green_max_size,
		cell_mask=cell_mask,
		threshold=green_threshold,
		threshold_method=green_threshold_method,
		threshold_percentile=green_threshold_percentile,
		gaussian_sigma=green_gaussian_sigma,
		local_bg_sigma=green_local_bg_sigma,
		fixed_threshold=green_fixed_threshold,
	)

	mito_mask, mito_skel = segment_mitochondria(
		ch1_in,
		min_size=mito_min_size,
		enhance=mito_enhance,
		threshold_method=mito_threshold_method,
		threshold_percentile=mito_threshold_percentile,
		threshold_local_block=mito_threshold_local_block,
		threshold_local_offset=mito_threshold_local_offset,
		threshold_sauvola_window=mito_threshold_sauvola_window,
		threshold_sauvola_k=mito_threshold_sauvola_k,
		threshold_niblack_window=mito_threshold_niblack_window,
		threshold_niblack_k=mito_threshold_niblack_k,
		contrast_sigma=mito_contrast_sigma,
		gaussian_sigma=mito_gaussian_sigma,
		prune_skeleton=mito_prune_skeleton,
		skeleton_min_length=mito_skeleton_min_length,
		min_mean_intensity=mito_min_mean_intensity,
		verbose=mito_verbose,
	)
	mito_metrics = compute_mito_metrics(mito_mask, mito_skel, cell_mask)

	stats = {
		"file": str(path.name),
		**orange_stats,
		"green_puncta_count": len(green_props),
		"green_total_area": float(np.sum(green_mask)),
		**{f"mito_{k}": v for k, v in mito_metrics.items()},
	}

	out_prefix = outdir / path.stem
	out_prefix.mkdir(parents=True, exist_ok=True)
	fig_path = out_prefix / "segmentation.png"
	# For visualization, show the channels masked by the cell so the overlay makes sense.
	visualize(ch1_in, ch2_in, ch3_in, ch4, cell_mask, green_mask, mito_mask, mito_skel, fig_path)

	csv_path = out_prefix / "stats.txt"
	with open(csv_path, "w") as f:
		for k, v in stats.items():
			f.write(f"{k}\t{v}\n")

	return stats, ch1, ch2, ch3, ch4, cell_mask, green_mask, mito_mask


def create_channel_montages(all_ch1, all_ch2, all_ch3, all_ch4, all_cell_masks, all_green_masks, all_mito_masks, outdir):
	"""Create montage images for channels and masks."""
	import math

	def make_montage(images, title, filename):
		n = len(images)
		cols = math.ceil(math.sqrt(n))
		rows = math.ceil(n / cols)
		fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
		if rows == 1 and cols == 1:
			axes = [axes]
		else:
			axes = axes.flatten()
		for i, img in enumerate(images):
			axes[i].imshow(img, cmap='gray')
			axes[i].axis('off')
		for i in range(n, len(axes)):
			axes[i].axis('off')
		fig.suptitle(title)
		fig.savefig(outdir / filename, dpi=150, bbox_inches='tight')
		plt.close(fig)

	make_montage(all_ch1, 'Mitochondria Channel (Ch1)', 'channel1_montage.png')
	make_montage(all_ch2, 'Orange/Red Channel (Ch2)', 'channel2_montage.png')
	make_montage(all_ch3, 'Green Channel (Ch3)', 'channel3_montage.png')
	make_montage(all_ch4, 'Cell Channel (Ch4)', 'channel4_montage.png')
	make_montage(all_cell_masks, 'Cell Masks', 'cell_masks_montage.png')
	make_montage(all_green_masks, 'Green Puncta Masks', 'green_masks_montage.png')
	make_montage(all_mito_masks, 'Mitochondria Masks', 'mito_masks_montage.png')


def main(argv: Optional[List[str]] = None):
	parser = argparse.ArgumentParser(description="Segment channels and compute morphology metrics")
	default_input = Path(__file__).resolve().parent / ".." / "Mito_Trogo"
	default_out = Path(__file__).resolve().parent / ".." / "Mito_Trogo" / "results"
	parser.add_argument(
		"--input-folder",
		required=False,
		type=Path,
		default=default_input,
		help=f"Folder with TIFFs (default: {default_input})",
	)
	parser.add_argument(
		"--outdir",
		required=False,
		type=Path,
		default=default_out,
		help=f"Output folder for results (default: {default_out})",
	)
	parser.add_argument("--cell-min-area", type=int, default=50_000, help="Minimum cell area in pixels")
	parser.add_argument("--cell-max-area", type=int, default=None, help="Maximum cell area in pixels (default: 50%% of image)")
	parser.add_argument("--cell-closing", type=int, default=10, help="Closing radius for cell mask")
	parser.add_argument(
		"--cell-threshold-method",
		type=str,
		choices=["otsu", "percentile", "local"],
		default="otsu",
		help="Thresholding method for cell segmentation (otsu, percentile, local)",
	)
	parser.add_argument(
		"--cell-threshold",
		type=float,
		default=None,
		help="Fixed intensity threshold for cell segmentation (overrides method)",
	)
	parser.add_argument(
		"--cell-threshold-percentile",
		type=float,
		default=99.0,
		help="Percentile for cell segmentation when using percentile threshold",
	)
	parser.add_argument(
		"--cell-threshold-local-block",
		type=int,
		default=51,
		help="Block size for local thresholding of cell segmentation",
	)
	parser.add_argument(
		"--cell-threshold-local-offset",
		type=float,
		default=0.0,
		help="Offset for local thresholding of cell segmentation",
	)
	parser.add_argument(
		"--cell-gaussian-sigma",
		type=float,
		default=0.0,
		help="Gaussian smoothing sigma applied before cell segmentation",
	)
	parser.add_argument(
		"--no-cell-keep-largest",
		action="store_false",
		dest="cell_keep_largest",
		help="Keep all cell-sized objects instead of only the largest",
	)
	parser.add_argument(
		"--cell-verbose",
		action="store_true",
		help="Print debug info for cell segmentation attempts",
	)
	parser.add_argument("--green-min-size", type=int, default=5, help="Minimum green puncta size")
	parser.add_argument("--green-max-size", type=int, default=5_000, help="Maximum green puncta size")
	parser.add_argument(
		"--green-threshold-method",
		type=str,
		choices=["otsu", "percentile"],
		default="percentile",
		help="Thresholding method for green puncta: otsu or percentile",
	)
	parser.add_argument(
		"--green-threshold",
		type=float,
		default=None,
		help="Fixed threshold for green puncta (overrides method)",
	)
	parser.add_argument(
		"--green-threshold-percentile",
		type=float,
		default=99.5,
		help="Percentile to use when threshold-method=percentile",
	)
	parser.add_argument(
		"--green-gaussian-sigma",
		type=float,
		default=1.0,
		help="Gaussian smoothing sigma before thresholding green channel",
	)
	parser.add_argument(
		"--green-local-bg-sigma",
		type=float,
		default=10.0,
		help="Sigma for Gaussian filter to estimate local background in green channel",
	)
	parser.add_argument(
		"--green-fixed-threshold",
		type=float,
		default=5.0,
		help="Fixed threshold value for green puncta after background subtraction (same for all images)",
	)
	parser.add_argument("--mito-min-size", type=int, default=50, help="Minimum mitochondria size in pixels")
	parser.add_argument(
		"--mito-enhance",
		type=str,
		choices=["none", "frangi"],
		default="none",
		help="Enhancement to apply before thresholding mitochondria",
	)
	parser.add_argument(
		"--mito-threshold-method",
		type=str,
		choices=["otsu", "percentile", "local", "sauvola", "niblack"],
		default="percentile",
		help="Threshold method for mitochondria segmentation",
	)
	parser.add_argument(
		"--mito-threshold-sauvola-window",
		type=int,
		default=25,
		help="Window size for Sauvola thresholding (only used when mito-threshold-method=sauvola)",
	)
	parser.add_argument(
		"--mito-threshold-sauvola-k",
		type=float,
		default=0.2,
		help="k parameter for Sauvola thresholding (only used when mito-threshold-method=sauvola)",
	)
	parser.add_argument(
		"--mito-threshold-niblack-window",
		type=int,
		default=25,
		help="Window size for Niblack thresholding (only used when mito-threshold-method=niblack)",
	)
	parser.add_argument(
		"--mito-threshold-niblack-k",
		type=float,
		default=0.2,
		help="k parameter for Niblack thresholding (only used when mito-threshold-method=niblack)",
	)
	parser.add_argument(
		"--mito-contrast-sigma",
		type=float,
		default=10.0,
		help="Sigma of Gaussian used to compute local background subtraction for contrast enhancement",
	)
	parser.add_argument(
		"--mito-skeleton-min-length",
		type=int,
		default=10,
		help="Minimum skeleton segment length (pixels); shorter segments are discarded",
	)
	parser.add_argument(
		"--mito-min-mean-intensity",
		type=float,
		default=0.15,
		help="Minimum mean normalized intensity for a mito object to be kept (0-1)",
	)
	parser.add_argument(
		"--mito-threshold-local-block",
		type=int,
		default=51,
		help="Block size for local thresholding (odd number, only used when mito-threshold-method=local)",
	)
	parser.add_argument(
		"--mito-threshold-local-offset",
		type=float,
		default=0.0,
		help="Offset for local thresholding (only used when mito-threshold-method=local)",
	)
	parser.add_argument(
		"--mito-threshold-percentile",
		type=float,
		default=90.0,
		help="Percentile threshold used when mito-threshold-method=percentile",
	)
	parser.add_argument(
		"--mito-gaussian-sigma",
		type=float,
		default=0.0,
		help="Gaussian smoothing sigma for mitochondria channel",
	)
	parser.add_argument(
		"--mito-prune-skeleton",
		action="store_true",
		help="Prune skeleton branches to isolate individual filaments",
	)
	parser.add_argument(
		"--mito-verbose",
		action="store_true",
		help="Print mitochondrial segmentation debug info (threshold values)",
	)
	args = parser.parse_args(argv)

	args.outdir.mkdir(parents=True, exist_ok=True)
	tifs = find_tifs(args.input_folder)
	if not tifs:
		print("No valid TIFFs found in", args.input_folder)
		print("If you see Git LFS pointer files (starting with 'version https://git-lfs.github.com'), run 'git lfs pull' before rerunning.")
		sys.exit(1)

	all_stats = []
	all_ch1 = []
	all_ch2 = []
	all_ch3 = []
	all_ch4 = []
	all_cell_masks = []
	all_green_masks = []
	all_mito_masks = []
	for tif in tifs:
		print(f"Processing: {tif}")
		stats, ch1, ch2, ch3, ch4, cell_mask, green_mask, mito_mask = process_file(
			tif,
			args.outdir,
			cell_min_area=args.cell_min_area,
			cell_max_area=args.cell_max_area,
			cell_closing=args.cell_closing,
			cell_threshold_method=args.cell_threshold_method,
			cell_threshold_value=args.cell_threshold,
			cell_threshold_percentile=args.cell_threshold_percentile,
			cell_threshold_local_block=args.cell_threshold_local_block,
			cell_threshold_local_offset=args.cell_threshold_local_offset,
			cell_gaussian_sigma=args.cell_gaussian_sigma,
			cell_keep_largest=args.cell_keep_largest,
			cell_verbose=args.cell_verbose,
			green_min_size=args.green_min_size,
			green_max_size=args.green_max_size,
			mito_min_size=args.mito_min_size,
			green_threshold_method=args.green_threshold_method,
			green_threshold=args.green_threshold,
			green_threshold_percentile=args.green_threshold_percentile,
			green_gaussian_sigma=args.green_gaussian_sigma,
			green_local_bg_sigma=args.green_local_bg_sigma,
			green_fixed_threshold=args.green_fixed_threshold,
			mito_enhance=args.mito_enhance,
			mito_threshold_method=args.mito_threshold_method,
			mito_threshold_percentile=args.mito_threshold_percentile,
			mito_threshold_local_block=args.mito_threshold_local_block,
			mito_threshold_local_offset=args.mito_threshold_local_offset,
			mito_gaussian_sigma=args.mito_gaussian_sigma,
			mito_prune_skeleton=args.mito_prune_skeleton,
			mito_min_mean_intensity=args.mito_min_mean_intensity,
			mito_verbose=args.mito_verbose,
		)
		all_stats.append(stats)
		all_ch1.append(ch1)
		all_ch2.append(ch2)
		all_ch3.append(ch3)
		all_ch4.append(ch4)
		all_cell_masks.append(cell_mask)
		all_green_masks.append(green_mask)
		all_mito_masks.append(mito_mask)

	summary_path = args.outdir / "summary.tsv"
	with open(summary_path, "w") as f:
		headers = list(all_stats[0].keys())
		f.write("\t".join(headers) + "\n")
		for s in all_stats:
			f.write("\t".join(str(s[h]) for h in headers) + "\n")

	# Create composite images
	create_channel_montages(all_ch1, all_ch2, all_ch3, all_ch4, all_cell_masks, all_green_masks, all_mito_masks, args.outdir)

	# Create plots comparing mito stats to green and red signals
	create_mito_comparison_plots(all_stats, args.outdir)


def create_mito_comparison_plots(all_stats, outdir):
	"""Create scatter plots of mito stats vs normalized green and red signals."""
	import matplotlib.pyplot as plt

	# Extract data
	green_per_area = [s['green_total_area'] / s['pixel_count'] for s in all_stats]
	red_per_area = [s['total_intensity'] / s['pixel_count'] for s in all_stats]
	mito_counts = [s['mito_num_mito_objects'] for s in all_stats]
	mito_mean_area = [s['mito_mean_area'] for s in all_stats]
	mito_total_length = [s['mito_total_skel_length'] for s in all_stats]

	# Plot mito count vs green per area
	fig, axes = plt.subplots(2, 2, figsize=(12, 10))
	axes[0,0].scatter(green_per_area, mito_counts)
	axes[0,0].set_xlabel('Green Signal per Cell Area')
	axes[0,0].set_ylabel('Number of Mitochondria')
	axes[0,0].set_title('Mito Count vs Green Signal Density')

	# Plot mito count vs red per area
	axes[0,1].scatter(red_per_area, mito_counts)
	axes[0,1].set_xlabel('Red Signal per Cell Area')
	axes[0,1].set_ylabel('Number of Mitochondria')
	axes[0,1].set_title('Mito Count vs Red Signal Density')

	# Plot mito mean area vs green per area
	axes[1,0].scatter(green_per_area, mito_mean_area)
	axes[1,0].set_xlabel('Green Signal per Cell Area')
	axes[1,0].set_ylabel('Mean Mito Area')
	axes[1,0].set_title('Mito Mean Area vs Green Signal Density')

	# Plot mito total length vs red per area
	axes[1,1].scatter(red_per_area, mito_total_length)
	axes[1,1].set_xlabel('Red Signal per Cell Area')
	axes[1,1].set_ylabel('Total Mito Skeleton Length')
	axes[1,1].set_title('Mito Total Length vs Red Signal Density')

	plt.tight_layout()
	fig.savefig(outdir / 'mito_comparison_plots.png', dpi=150)
	plt.close(fig)


if __name__ == "__main__":
	main()

