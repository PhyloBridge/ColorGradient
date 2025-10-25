"""
VERSION: 2025-10-26
AUTHOR: ColorGradient Contributor(s), © 2025. All rights reserved.
LICENSE: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

Usage:
from colorgradient import plot_gradient_grid
fig, axes = plot_gradient_grid(cell_data, color_schema)
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Patch
from typing import Union, List, Tuple, Dict, Any


'''
////////////////////////////////////////////////////////////

	SECTION-1: Gradient Generation

////////////////////////////////////////////////////////////
'''
DEFAULT_COLOR_SCHEMA = {
	0: '#F44336',
	50: '#FB7B8E',
	80: '#1F2F98',
	90: '#2973B2',
	95: '#4E71FF',
	97.5: '#4ED7F1',
	100: '#38E54D'
}

def create_circular_gradient(data: Union[List, np.ndarray], resolution: int = 100) -> np.ndarray:
	"""Create circular gradient array from 1D data distribution with enhanced percentile-based radial interpolation."""
	if data is None or len(data) == 0:
		raise ValueError("Data cannot be empty")
	if not isinstance(data, np.ndarray):
		data = np.array(data)
	if data.ndim != 1:
		raise ValueError(f"Data must be 1D array, got {data.ndim}D")
	data_clean = data[np.isfinite(data)]
	if data_clean.size == 0:
		return np.full((resolution, resolution), np.nan, dtype=float)
	sorted_data = np.sort(data_clean)
	percentiles = np.linspace(0, 100, resolution)
	radial_values = np.percentile(sorted_data, 100 - percentiles)
	y_coords, x_coords = np.ogrid[-1:1:resolution*1j, -1:1:resolution*1j]
	radial_distance = np.sqrt(x_coords**2 + y_coords**2)
	radial_distance = np.clip(radial_distance, 0, 1)
	radial_indices = (radial_distance * (resolution - 1)).astype(int)
	radial_indices = np.clip(radial_indices, 0, resolution - 1)
	gradient_array = radial_values[radial_indices]
	return gradient_array


'''
////////////////////////////////////////////////////////////

	SECTION-2: Border Highlighting Helper Functions

////////////////////////////////////////////////////////////
'''
def compute_grid_statistics(grid_data: Dict[Tuple[int, int], Union[List, np.ndarray]]) -> Dict[str, np.ndarray]:
	"""Compute statistics (mean, median, min, max, p25, p75) for nested grid data."""
	if not grid_data:
		return {}
	all_rows = sorted(set(key[0] for key in grid_data.keys()))
	all_cols = sorted(set(key[1] for key in grid_data.keys()))
	num_rows, num_cols = len(all_rows), len(all_cols)
	row_map = {r: i for i, r in enumerate(all_rows)}
	col_map = {c: i for i, c in enumerate(all_cols)}
	stats = {
		'mean': np.full((num_rows, num_cols), np.nan),
		'median': np.full((num_rows, num_cols), np.nan),
		'min': np.full((num_rows, num_cols), np.nan),
		'max': np.full((num_rows, num_cols), np.nan),
		'p25': np.full((num_rows, num_cols), np.nan),
		'p75': np.full((num_rows, num_cols), np.nan)
	}
	for (row_key, col_key), data_values in grid_data.items():
		if not isinstance(data_values, np.ndarray):
			data_values = np.array(data_values)
		data_clean = data_values[np.isfinite(data_values)]
		if data_clean.size == 0:
			continue
		row_idx, col_idx = row_map[row_key], col_map[col_key]
		stats['mean'][row_idx, col_idx] = np.mean(data_clean)
		stats['median'][row_idx, col_idx] = np.median(data_clean)
		stats['min'][row_idx, col_idx] = np.min(data_clean)
		stats['max'][row_idx, col_idx] = np.max(data_clean)
		stats['p25'][row_idx, col_idx] = np.percentile(data_clean, 25)
		stats['p75'][row_idx, col_idx] = np.percentile(data_clean, 75)
	return stats

def find_best_cells_in_grid(stats_grids: Dict[str, np.ndarray], border_configs: Dict[str, Dict]) -> Dict[str, Tuple[int, int]]:
	"""Find local best cells for each enabled border type in a single grid."""
	best_cells = {}
	for border_type, config in border_configs.items():
		if not config.get('enabled', False):
			continue
		if not border_type.startswith('local_'):
			continue
		parts = border_type.split('_')
		stat = '_'.join(parts[2:])
		if stat not in stats_grids:
			continue
		grid = stats_grids[stat]
		if np.all(np.isnan(grid)):
			continue
		idx = np.nanargmax(grid)
		row_idx, col_idx = np.unravel_index(idx, grid.shape)
		best_cells[border_type] = (row_idx, col_idx)
	return best_cells

def find_all_best_cells_global(all_grids: Dict[int, Dict[str, np.ndarray]], border_configs: Dict[str, Dict]) -> Dict[str, Tuple[int, int, int]]:
	"""Find global best cells across all subplots."""
	global_bests = {}
	for border_type, config in border_configs.items():
		if not config.get('enabled', False):
			continue
		if not border_type.startswith('global_'):
			continue
		parts = border_type.split('_')
		stat = '_'.join(parts[2:])
		best_val = -np.inf
		best_loc = None
		for subplot_idx, stats_grids in all_grids.items():
			if stat not in stats_grids:
				continue
			grid = stats_grids[stat]
			if np.all(np.isnan(grid)):
				continue
			idx = np.nanargmax(grid)
			row_idx, col_idx = np.unravel_index(idx, grid.shape)
			val = grid[row_idx, col_idx]
			if val > best_val:
				best_val = val
				best_loc = (subplot_idx, row_idx, col_idx)
		if best_loc is not None:
			global_bests[border_type] = best_loc
	return global_bests


'''
////////////////////////////////////////////////////////////

	SECTION-3: Color Schema System

////////////////////////////////////////////////////////////
'''
def create_color_map(color_schema: Dict[float, str]) -> Tuple:
	"""Create matplotlib colormap from color schema dictionary."""
	if not color_schema:
		raise ValueError("color_schema cannot be empty")
	sorted_keys = sorted(color_schema.keys())
	vmin, vmax = sorted_keys[0], sorted_keys[-1]
	colors = [color_schema[key] for key in sorted_keys]
	positions = [(key - vmin) / (vmax - vmin) for key in sorted_keys]
	cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', list(zip(positions, colors)))
	return cmap, vmin, vmax

def validate_data_range(data: Union[List, np.ndarray], color_schema: Dict[float, str], cell_name: str = "Cell"):
	"""Validate that data values are within color schema range."""
	if not isinstance(data, np.ndarray):
		data = np.array(data)
	data_min = np.min(data)
	data_max = np.max(data)
	schema_min = min(color_schema.keys())
	schema_max = max(color_schema.keys())
	if data_min < schema_min:
		print(f"Warning: {cell_name} has minimum value {data_min:.2f} below color schema minimum {schema_min:.2f}")
	if data_max > schema_max:
		print(f"Warning: {cell_name} has maximum value {data_max:.2f} exceeding color schema maximum {schema_max:.2f}. Consider redefining schema.")


'''
////////////////////////////////////////////////////////////

	SECTION-4: Grid Layout System

////////////////////////////////////////////////////////////
'''
def calculate_grid_layout(num_cells: int, rows: int = None, cols: int = None, max_rows: int = None, max_cols: int = None) -> Tuple[int, int]:
	"""Calculate optimal grid layout for given number of cells."""
	if num_cells <= 0:
		raise ValueError("Number of cells must be positive")
	if rows is not None and cols is not None:
		if rows * cols < num_cells:
			raise ValueError(f"Fixed grid {rows}x{cols} cannot fit {num_cells} cells")
		return rows, cols
	if rows is not None:
		calculated_cols = int(np.ceil(num_cells / rows))
		if max_cols is not None and calculated_cols > max_cols:
			raise ValueError(f"Cannot fit {num_cells} cells in {rows} rows with max_cols={max_cols}")
		return rows, calculated_cols
	if cols is not None:
		calculated_rows = int(np.ceil(num_cells / cols))
		if max_rows is not None and calculated_rows > max_rows:
			raise ValueError(f"Cannot fit {num_cells} cells in {cols} columns with max_rows={max_rows}")
		return calculated_rows, cols
	default_cols = min(3, num_cells)
	if max_cols is not None:
		default_cols = min(default_cols, max_cols)
	calculated_rows = int(np.ceil(num_cells / default_cols))
	if max_rows is not None and calculated_rows > max_rows:
		default_cols = int(np.ceil(num_cells / max_rows))
		calculated_rows = max_rows
	return calculated_rows, default_cols

def create_square_subplots(rows: int, cols: int, figsize_per_cell: float = 4.0) -> Tuple:
	"""Create matplotlib figure with square subplots."""
	figsize = (cols * figsize_per_cell, rows * figsize_per_cell)
	fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
	for ax in axes.flat:
		ax.set_aspect('equal', adjustable='box')
	return fig, axes


'''
////////////////////////////////////////////////////////////

	SECTION-5: Nested Grid Support

////////////////////////////////////////////////////////////
'''
def draw_nested_grid_subplot(ax, cell_config: Dict, color_schema: Dict, cmap, vmin: float, vmax: float, resolution: int = 100, show_values: bool = False, bold_titles: bool = False, borders_to_draw: Dict[str, Tuple[int, int]] = None, border_configs: Dict[str, Dict] = None):
	"""Draw a nested grid subplot where each cell contains circular gradients from data distributions."""
	grid_data = cell_config.get('grid_data')
	if not grid_data:
		ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
		ax.set_xticks([])
		ax.set_yticks([])
		return
	all_rows = sorted(set(key[0] for key in grid_data.keys()))
	all_cols = sorted(set(key[1] for key in grid_data.keys()))
	num_rows, num_cols = len(all_rows), len(all_cols)
	row_map = {r: i for i, r in enumerate(all_rows)}
	col_map = {c: i for i, c in enumerate(all_cols)}
	base_grid = np.full((num_rows, num_cols), np.nan)
	for (row_key, col_key), data_values in grid_data.items():
		if not isinstance(data_values, np.ndarray):
			data_values = np.array(data_values)
		data_clean = data_values[np.isfinite(data_values)]
		if data_clean.size > 0:
			row_idx, col_idx = row_map[row_key], col_map[col_key]
			base_grid[row_idx, col_idx] = np.mean(data_clean)
	cell_resolution = max(50, resolution // max(num_rows, num_cols))
	for (row_key, col_key), data_values in grid_data.items():
		row_idx, col_idx = row_map[row_key], col_map[col_key]
		if not isinstance(data_values, np.ndarray):
			data_values = np.array(data_values)
		data_clean = data_values[np.isfinite(data_values)]
		if data_clean.size == 0:
			continue
		gradient = create_circular_gradient(data_clean, resolution=cell_resolution)
		epsilon = 0.001
		cell_extent = [col_idx - 0.5 - epsilon, col_idx + 0.5 + epsilon, row_idx + 0.5 + epsilon, row_idx - 0.5 - epsilon]
		ax.imshow(gradient, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', extent=cell_extent, interpolation='bilinear', zorder=2)
	ax.set_xticks(np.arange(num_cols))
	ax.set_yticks(np.arange(num_rows))
	row_labels = cell_config.get('row_labels', [str(r) for r in all_rows])
	col_labels = cell_config.get('col_labels', [str(c) for c in all_cols])
	ax.set_xticklabels(col_labels)
	ax.set_yticklabels(row_labels)
	if 'xlabel' in cell_config:
		ax.set_xlabel(cell_config['xlabel'], fontsize=10)
	if 'ylabel' in cell_config:
		ax.set_ylabel(cell_config['ylabel'], fontsize=10)
	if 'title' in cell_config:
		title_kwargs = {'fontsize': 12}
		if bold_titles:
			title_kwargs['fontweight'] = 'bold'
		ax.set_title(cell_config['title'], **title_kwargs)
	ax.set_xlim(-0.5, num_cols - 0.5)
	ax.set_ylim(num_rows - 0.5, -0.5)
	if borders_to_draw and border_configs:
		border_inset_step = 0.05
		border_linewidth = 2
		for row_idx in range(num_rows):
			for col_idx in range(num_cols):
				cell_borders = []
				for border_type, coord in borders_to_draw.items():
					if coord == (row_idx, col_idx):
						cell_borders.append(border_type)
				if cell_borders:
					sorted_borders = sorted(cell_borders, key=lambda bt: list(border_configs.keys()).index(bt) if bt in border_configs else 999)
					for inset_index, border_type in enumerate(sorted_borders):
						inset_amount = inset_index * border_inset_step
						color = border_configs[border_type]['color']
						rect = Rectangle((col_idx - 0.5 + inset_amount, row_idx - 0.5 + inset_amount), 1 - 2 * inset_amount, 1 - 2 * inset_amount, fill=False, edgecolor=color, linewidth=border_linewidth, zorder=20)
						ax.add_patch(rect)
	if show_values:
		font_size = max(6, 12 - max(num_rows, num_cols))
		for row_idx in range(num_rows):
			for col_idx in range(num_cols):
				if not np.isnan(base_grid[row_idx, col_idx]):
					mean_val = base_grid[row_idx, col_idx]
					text_obj = ax.text(col_idx, row_idx, f'{mean_val:.1f}', ha='center', va='center', color='white', fontsize=font_size, weight='bold', zorder=25)
					try:
						text_obj.set_path_effects([pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
					except:
						pass


'''
////////////////////////////////////////////////////////////

	SECTION-6: High-Level API

////////////////////////////////////////////////////////////
'''
def plot_gradient_grid(
	cell_data: List[Dict[str, Any]],
	color_schema: Dict[float, str] = None,
	rows: int = None,
	cols: int = None,
	max_rows: int = None,
	max_cols: int = None,
	figsize_per_cell: float = 4.0,
	resolution: int = 500,
	highlight_borders: Dict[str, Dict] = None,
	show_values: bool = False,
	bold_titles: bool = False,
	suptitle: Dict[str, Any] = None,
	copyright_text: str = None,
) -> Tuple:
	"""Create grid plot with circular gradient visualization for each cell."""
	if not cell_data or len(cell_data) == 0:
		raise ValueError("cell_data cannot be empty")
	if color_schema is None:
		color_schema = DEFAULT_COLOR_SCHEMA
	if highlight_borders is None:
		highlight_borders = {}
	enabled_borders = {k: v for k, v in highlight_borders.items() if v.get('enabled', False)}
	if len(enabled_borders) > 4:
		warnings.warn(f"More than 4 borders enabled ({len(enabled_borders)}). Visual clarity may be compromised. Proceeding anyway.")
	num_cells = len(cell_data)
	grid_rows, grid_cols = calculate_grid_layout(num_cells, rows=rows, cols=cols, max_rows=max_rows, max_cols=max_cols)
	fig, axes = create_square_subplots(grid_rows, grid_cols, figsize_per_cell=figsize_per_cell)
	cmap, vmin, vmax = create_color_map(color_schema)
	all_stats_grids = {}
	for cell_idx, cell in enumerate(cell_data):
		if 'grid_data' in cell:
			stats_grids = compute_grid_statistics(cell['grid_data'])
			all_stats_grids[cell_idx] = stats_grids
	# Filter out subplots marked for exclusion before finding global bests
	grids_for_best_calc = {
		idx: stats for idx, stats in all_stats_grids.items()
		if not cell_data[idx].get('exclude_from_best', False)
	}
	global_bests = find_all_best_cells_global(grids_for_best_calc, highlight_borders)
	cell_idx = 0
	for row in range(grid_rows):
		for col in range(grid_cols):
			ax = axes[row, col]
			if cell_idx < num_cells:
				cell = cell_data[cell_idx]
				if 'grid_data' in cell:
					stats_grids = all_stats_grids.get(cell_idx, {})
					# Only compute local bests if NOT excluded
					if not cell.get('exclude_from_best', False):
						local_bests = find_best_cells_in_grid(stats_grids, highlight_borders)
					else:
						local_bests = {}
					borders_for_cell = local_bests.copy()
					for border_type, (subplot_idx, r_idx, c_idx) in global_bests.items():
						if subplot_idx == cell_idx:
							borders_for_cell[border_type] = (r_idx, c_idx)
					draw_nested_grid_subplot(ax, cell, color_schema, cmap, vmin, vmax, resolution, show_values=show_values, bold_titles=bold_titles, borders_to_draw=borders_for_cell, border_configs=highlight_borders)
				else:
					data = cell.get('data')
					if data is None:
						raise ValueError(f"Cell {cell_idx} missing required 'data' field")
					cell_name = cell.get('title', f"Cell ({row},{col})")
					validate_data_range(data, color_schema, cell_name)
					gradient = create_circular_gradient(data, resolution=resolution)
					im = ax.imshow(gradient, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', origin='upper')
					if 'title' in cell:
						title_kwargs = {'fontsize': 12}
						if bold_titles:
							title_kwargs['fontweight'] = 'bold'
						ax.set_title(cell['title'], **title_kwargs)
					if 'xlabel' in cell:
						ax.set_xlabel(cell['xlabel'], fontsize=10)
					if 'ylabel' in cell:
						ax.set_ylabel(cell['ylabel'], fontsize=10)
					ax.set_xticks([])
					ax.set_yticks([])
					if show_values:
						mean_val = np.mean(data)
						text_obj = ax.text(0.5, 0.5, f'{mean_val:.1f}', transform=ax.transAxes, ha='center', va='center', color='white', fontsize=14, weight='bold', zorder=10)
						try:
							text_obj.set_path_effects([pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
						except:
							pass
				cell_idx += 1
			else:
				ax.axis('off')
	if suptitle:
		title_text = suptitle.get('title', '')
		if title_text:
			fontsize = suptitle.get('fontsize', 14)
			fontweight = 'bold' if suptitle.get('bold', False) else 'normal'
			num_lines = title_text.count('\n') + 1
			y_position = suptitle['y_override'] if 'y_override' in suptitle else 0.95 - (num_lines - 1) * 0.02
			fig.suptitle(title_text, fontsize=fontsize, fontweight=fontweight, y=y_position)
			top_margin = 0.96 - (num_lines - 1) * 0.03
		else:
			top_margin = 0.96
	else:
		top_margin = 0.96
	if enabled_borders:
		legend_handles = []
		for border_type, config in highlight_borders.items():
			if config.get('enabled', False):
				label_map = {
					'local_best_mean': 'Local Best (Mean)',
					'local_best_median': 'Local Best (Median)',
					'local_best_min': 'Local Best (Min)',
					'local_best_max': 'Local Best (Max)',
					'local_best_p25': 'Local Best (25th percentile)',
					'local_best_p75': 'Local Best (75th percentile)',
					'global_best_mean': 'Global Best (Mean)',
					'global_best_median': 'Global Best (Median)',
					'global_best_min': 'Global Best (Min)',
					'global_best_max': 'Global Best (Max)',
					'global_best_p25': 'Global Best (25th percentile)',
					'global_best_p75': 'Global Best (75th percentile)'
				}
				label = config.get('override', label_map.get(border_type, border_type))
				patch = Patch(facecolor='none', edgecolor=config['color'], label=label, linewidth=2)
				legend_handles.append(patch)
		if legend_handles:
			fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
					bbox_to_anchor=(0.5, -0.01), frameon=True, shadow=True, fancybox=True)
			plt.tight_layout(rect=[0, 0.02, 1, top_margin])
		else:
			plt.tight_layout(rect=[0, 0, 1, top_margin])
	else:
		plt.tight_layout(rect=[0, 0, 1, top_margin])
	if copyright_text:
		fig.text(0.99, 0.01, copyright_text, fontsize=8, color='#CCCCCC', ha='right', va='bottom')
	return fig, axes