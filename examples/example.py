"""
VERSION: 2025-10-26
AUTHOR: ColorGradient Contributor(s), © 2025. All rights reserved.
LICENSE: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
mpl.rcParams["svg.hashsalt"] = "42"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from colorgradient import plot_gradient_grid, DEFAULT_COLOR_SCHEMA
from functools import partial
print = partial(print, flush=True)


BLUE_CYAN_SCHEMA = {
	0: '#1E3A8A',		# Dark blue (poor performance)
	100: '#00FFFF'		# Cyan (excellent performance)
}

def example1_basic_distributions():
	"""Example 1: Compare different statistical distributions with blue to cyan schema"""
	print(f"Example 1: Basic Distribution Comparison")
	np.random.seed(42)
	cell_data = [
		{'data': np.random.normal(75, 10, 100), 'title': 'Normal Distribution\nnp.random.normal(75, 10, 100)'},
		{'data': np.random.exponential(20, 80), 'title': 'Exponential Distribution\nnp.random.exponential(20, 80)'},
		{'data': np.random.uniform(0, 100, 100), 'title': 'Uniform Distribution\nnp.random.uniform(0, 100, 100)'},
		{'data': [10, 30, 50, 70, 90], 'title': 'Sparse Data\n[10, 30, 50, 70, 90]'}
	]
	fig, axes = plot_gradient_grid(
					cell_data=cell_data,
					color_schema=BLUE_CYAN_SCHEMA,
					rows=2,
					cols=2,
					figsize_per_cell=4,
					suptitle={
						'title': 'Distribution Comparison',
						'fontsize': 14,
						'bold': True
					}
				)
	# Move all subplots left to make room for colorbar
	plt.subplots_adjust(right=0.82)
	# Calculate proper colorbar position to match subplot heights
	subplot_positions = axes[0, 0].get_position()
	subplot_bottom = subplot_positions.y0
	subplot_top = axes[0, 0].get_position().y1
	subplot_height = subplot_top - subplot_bottom
	row_spacing = axes[0, 0].get_position().y0 - axes[1, 0].get_position().y1
	total_height = 2 * subplot_height + row_spacing
	colorbar_bottom = axes[1, 0].get_position().y0		# Bottom-left subplot
	colorbar_axis = fig.add_axes([0.85, colorbar_bottom, 0.02, total_height])
	# Create colormap
	colormap = mcolors.LinearSegmentedColormap.from_list('darkblue_to_cyan', ['#1E3A8A', '#00FFFF'])
	norm = mcolors.Normalize(vmin=0, vmax=100)
	colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), cax=colorbar_axis)
	colorbar.set_label('Performance Score', rotation=270, labelpad=20)
	# Saving
	plt.savefig('example1_basic_distributions.svg', dpi=300, bbox_inches='tight', metadata={'Date': None})
	plt.close(fig)
	print(f"-> Saved: example1_basic_distributions.svg")

def example2_hyperparameter_grid():
	"""Example 2: Hyperparameter search grid (3x3) for a single model"""
	print(f"\nExample 2: Hyperparameter Search Grid")
	np.random.seed(42)
	epochs = [3, 6, 9]
	learning_rates = [1e-3, 1e-4, 1e-5]
	grid_data = {}
	for learning_rate_index, learning_rate in enumerate(learning_rates):
		for epoch_index, epoch in enumerate(epochs):
			base_performance = 60 + epoch * 2 - learning_rate_index * 5
			performance = np.random.normal(base_performance, 5 + learning_rate_index * 2, 50)
			grid_data[(learning_rate_index, epoch_index)] = np.clip(performance, 0, 100)
	cell_data = [{
		'grid_data': grid_data,
		'title': 'Model Name',
		'row_labels': [f'{lr:.0e}' for lr in learning_rates],
		'col_labels': [str(e) for e in epochs],
		'xlabel': 'Epochs',
		'ylabel': 'Learning Rate'
	}]
	fig, axes = plot_gradient_grid(
		cell_data,
		DEFAULT_COLOR_SCHEMA,
		rows=1,
		cols=1,
		figsize_per_cell=6,
		show_values=True
	)
	fig.suptitle('Hyperparameter Grid with Value Annotations', fontsize=14)
	output_directory = os.path.dirname(__file__)
	plt.savefig(os.path.join(output_directory, 'example2_hyperparameter_grid.svg'), dpi=300, bbox_inches='tight', metadata={'Date': None})
	plt.close(fig)
	print(f"-> Saved: example2_hyperparameter_grid.svg")

if __name__ == '__main__':
	print("=" * 60)
	print("ColorGradient Examples: Visualizing Data Distributions")
	print("=" * 60)
	example1_basic_distributions()
	example2_hyperparameter_grid()
	print("\n" + "=" * 60)
	print("All examples generated successfully!")
	print("=" * 60)