"""
VERSION: 2025-10-26
AUTHOR: ColorGradient Contributor(s), © 2025. All rights reserved.
LICENSE: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["svg.hashsalt"] = "42"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from colorgradient import plot_gradient_grid, DEFAULT_COLOR_SCHEMA
from functools import partial
print = partial(print, flush=True)


def generate_model_data(model_idx, epochs, learning_rates, seed=42):
	"""Generate synthetic hyperparameter search data for a model"""
	np.random.seed(seed + model_idx)
	grid_data = {}
	for lr_idx, lr in enumerate(learning_rates):
		for epoch_idx, epoch in enumerate(epochs):
			base_performance = 60 + model_idx * 5
			epoch_effect = epoch * 2
			lr_effect = (2 - lr_idx) * 3
			noise_scale = 8 + lr_idx * 2
			n_samples = 50
			performance_data = np.random.normal(base_performance + epoch_effect + lr_effect, noise_scale, n_samples)
			performance_data = np.clip(performance_data, 0, 100)
			grid_data[(lr_idx, epoch_idx)] = performance_data
	return grid_data

def example1():
	"""Generate hyperparameter comparison without annotations"""
	print("Example 1: Hyperparameter Grid (No Annotations)")
	models = ['Model-1', 'Model-2', 'Model-3', 'Model-4', 'Model-5']
	epochs = [3, 6, 9]
	learning_rates = [1e-3, 1e-4, 1e-5]
	subplot_data = []
	for model_idx, model_name in enumerate(models):
		grid_data = generate_model_data(model_idx, epochs, learning_rates)
		subplot_config = {
			'grid_data': grid_data,
			'title': f'{model_name}',
			'row_labels': [f'{lr:.0e}' for lr in learning_rates],
			'col_labels': [str(epoch) for epoch in epochs],
			'xlabel': 'Epochs',
			'ylabel': 'Learning Rate'
		}
		subplot_data.append(subplot_config)
	color_schema_grid = {}
	schema_values = [[75.0, 90.0, 100.0], [70.0, 85.0, 97.5], [65.0, 80.0, 95.0]]
	for i in range(3):
		for j in range(3):
			color_schema_grid[(i, j)] = [schema_values[i][j]]
	color_schema_config = {
		'grid_data': color_schema_grid,
		'title': 'Color Schema',
		'row_labels': [],
		'col_labels': [],
		'exclude_from_best': True
	}
	subplot_data.append(color_schema_config)
	fig, axes = plot_gradient_grid(
		subplot_data,
		DEFAULT_COLOR_SCHEMA,
		rows=2,
		cols=3,
		figsize_per_subplot=4,
		show_values=False
	)
	fig.suptitle('Model Comparison: Hyperparameter Search Results', fontsize=14)
	output_dir = os.path.dirname(__file__)
	plt.savefig(os.path.join(output_dir, 'example_hyperparameter_grid_1.svg'), dpi=300, bbox_inches='tight', metadata={'Date': None})
	plt.savefig(os.path.join(output_dir, 'example_hyperparameter_grid_1.png'), dpi=300, bbox_inches='tight', metadata={'Software': None})
	plt.close(fig)
	print("-> Saved: example_hyperparameter_grid_1.svg")
	print("-> Saved: example_hyperparameter_grid_1.png")

def example2_annotated():
	"""Generate hyperparameter comparison with value annotations"""
	print("\nExample 2: Annotated Hyperparameter Grid (With Mean Values)")
	models = ['Model-1', 'Model-2', 'Model-3', 'Model-4', 'Model-5']
	epochs = [3, 6, 9]
	learning_rates = [1e-3, 1e-4, 1e-5]
	subplot_data = []
	for model_idx, model_name in enumerate(models):
		grid_data = generate_model_data(model_idx, epochs, learning_rates)
		subplot_config = {
			'grid_data': grid_data,
			'title': f'{model_name}',
			'row_labels': [f'{lr:.0e}' for lr in learning_rates],
			'col_labels': [str(epoch) for epoch in epochs],
			'xlabel': 'Epochs',
			'ylabel': 'Learning Rate'
		}
		subplot_data.append(subplot_config)
	color_schema_grid = {}
	schema_values = [[75.0, 90.0, 100.0], [70.0, 85.0, 97.5], [65.0, 80.0, 95.0]]
	for i in range(3):
		for j in range(3):
			color_schema_grid[(i, j)] = [schema_values[i][j]]
	color_schema_config = {
		'grid_data': color_schema_grid,
		'title': 'Color Schema',
		'row_labels': [],
		'col_labels': [],
		'exclude_from_best': True
	}
	subplot_data.append(color_schema_config)
	fig, axes = plot_gradient_grid(
		subplot_data,
		DEFAULT_COLOR_SCHEMA,
		rows=2,
		cols=3,
		figsize_per_subplot=4,
		show_values=True
	)
	fig.suptitle('Model Comparison with Mean Annotations', fontsize=14)
	output_dir = os.path.dirname(__file__)
	plt.savefig(os.path.join(output_dir, 'example_hyperparameter_grid_2.svg'), dpi=300, bbox_inches='tight', metadata={'Date': None})
	plt.savefig(os.path.join(output_dir, 'example_hyperparameter_grid_2.png'), dpi=300, bbox_inches='tight', metadata={'Software': None})
	plt.close(fig)
	print("-> Saved: example_hyperparameter_grid_2.svg")
	print("-> Saved: example_hyperparameter_grid_2.png")

def example3_border_highlighting():
	"""Generate hyperparameter comparison with local/global best border highlighting"""
	print("\nExample 3: Border Highlighting (Local Best + Global Best Mean)")
	models = ['Model-1', 'Model-2', 'Model-3', 'Model-4', 'Model-5']
	epochs = [3, 6, 9]
	learning_rates = [1e-3, 1e-4, 1e-5]
	subplot_data = []
	for model_idx, model_name in enumerate(models):
		grid_data = generate_model_data(model_idx, epochs, learning_rates)
		subplot_config = {
			'grid_data': grid_data,
			'title': f'{model_name}',
			'row_labels': [f'{lr:.0e}' for lr in learning_rates],
			'col_labels': [str(epoch) for epoch in epochs],
			'xlabel': 'Epochs',
			'ylabel': 'Learning Rate'
		}
		subplot_data.append(subplot_config)
	color_schema_grid = {}
	schema_values = [[75.0, 90.0, 100.0], [70.0, 85.0, 97.5], [65.0, 80.0, 95.0]]
	for i in range(3):
		for j in range(3):
			color_schema_grid[(i, j)] = [schema_values[i][j]]
	color_schema_config = {
		'grid_data': color_schema_grid,
		'title': 'Color Schema',
		'row_labels': [],
		'col_labels': [],
		'exclude_from_best': True
	}
	subplot_data.append(color_schema_config)
	highlight_borders = {
		'local_best_mean': {'enabled': True, 'color': '#08CB00', 'override': 'Local best (Average)'},
		'local_best_max': {'enabled': True, 'color': '#0FFFFF'},
		'global_best_mean': {'enabled': True, 'color': '#F7A8C4', 'override': 'Global best (Average)'},
		'global_best_median': {'enabled': True, 'color': '#FFD65A'}
	}
	suptitle_config = {
		'title': 'Hyperparameter Grid Comparison\nAcross 5 Models and Various Configurations',
		'fontsize': 16,
		'bold': False,
		#'y_override': 0.97	# Optional manual override: recommended range 0.90 (very tight) to 0.97 (a lot of white space)
	}
	fig, axes = plot_gradient_grid(
		subplot_data, 
		DEFAULT_COLOR_SCHEMA, 
		rows=2, 
		cols=3, 
		figsize_per_subplot=4,
		resolution=500,
		highlight_borders=highlight_borders,
		show_values=True,
		bold_titles=False,
		suptitle=suptitle_config,
		copyright_text='© 2025 ColorGradient Contributors',
	)
	output_dir = os.path.dirname(__file__)
	plt.savefig(os.path.join(output_dir, 'example_hyperparameter_grid_3.svg'), dpi=300, bbox_inches='tight', metadata={'Date': None})
	plt.savefig(os.path.join(output_dir, 'example_hyperparameter_grid_3.png'), dpi=300, bbox_inches='tight', metadata={'Software': None})
	plt.close(fig)
	print("-> Saved: example_hyperparameter_grid_3.svg")
	print("-> Saved: example_hyperparameter_grid_3.png")
	print("   Features: Local Best (green), Global Best Mean (pink), Global Best Median (yellow)")
	print("   Copyright text displayed at bottom right")

if __name__ == '__main__':
	print("=" * 70)
	print("ColorGradient Hyperparameter Grid Examples")
	print("=" * 70)
	example1()
	example2_annotated()
	example3_border_highlighting()
	print("\n" + "=" * 70)
	print("All hyperparameter grid examples generated successfully!")
	print("=" * 70)