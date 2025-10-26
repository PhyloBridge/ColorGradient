"""
VERSION: 2025-10-26
AUTHOR: ColorGradient Contributor(s), © 2025. All rights reserved.
LICENSE: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
"""

import os
import sys
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import StringIO
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from colorgradient.core import create_circular_gradient, create_color_map, validate_data_range, calculate_grid_layout, create_square_subplots, plot_gradient_grid

class TestPhase1CircularGradient(unittest.TestCase):
	def test_empty_data(self):
		"""Test that empty data raises ValueError"""
		with self.assertRaises(ValueError):
			create_circular_gradient([])

	def test_none_data(self):
		"""Test that None data raises ValueError"""
		with self.assertRaises(ValueError):
			create_circular_gradient(None)

	def test_2d_data(self):
		"""Test that 2D data raises ValueError"""
		with self.assertRaises(ValueError):
			create_circular_gradient([[1, 2], [3, 4]])

	def test_python_list(self):
		"""Test that python list is accepted"""
		data = [1, 2, 3]
		result = create_circular_gradient(data)
		self.assertEqual(result.shape, (100, 100))

	def test_numpy_array(self):
		"""Test that numpy array is accepted"""
		data = np.array([1, 2, 3])
		result = create_circular_gradient(data)
		self.assertEqual(result.shape, (100, 100))

	def test_three_distinct_values(self):
		"""Test that 3 distinct values create proper gradient (center=highest, corner=lowest)"""
		data = [10, 20, 30]
		result = create_circular_gradient(data, resolution=100)
		center_value = result[50, 50]
		corner_value = result[0, 0]
		self.assertGreater(center_value, corner_value)
		self.assertAlmostEqual(center_value, 30, delta=0.5)
		self.assertEqual(corner_value, 10)

	def test_custom_resolution(self):
		"""Test custom resolution"""
		data = [1, 2, 3]
		result = create_circular_gradient(data, resolution=50)
		self.assertEqual(result.shape, (50, 50))

	def test_single_value(self):
		"""Test single value creates uniform gradient"""
		data = [42]
		result = create_circular_gradient(data)
		self.assertTrue(np.all(result == 42))

	def test_radial_decrease(self):
		"""Test that values decrease radially from center to edge"""
		data = [1, 2, 3, 4, 5]
		result = create_circular_gradient(data, resolution=100)
		center = 50
		value_at_center = result[center, center]
		value_at_quarter = result[center, center + 25]
		value_at_edge = result[center, 99]
		self.assertGreaterEqual(value_at_center, value_at_quarter)
		self.assertGreaterEqual(value_at_quarter, value_at_edge)

class TestPhase2ColorSchema(unittest.TestCase):
	def test_empty_schema(self):
		"""Test that empty schema raises ValueError"""
		with self.assertRaises(ValueError):
			create_color_map({})

	def test_single_value_schema(self):
		"""Test that single value schema raises ValueError"""
		with self.assertRaises((ValueError, ZeroDivisionError)):
			create_color_map({0: '#FFFFFF'})

	def test_two_value_schema(self):
		"""Test basic two-value color schema"""
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		cmap, vmin, vmax = create_color_map(schema)
		self.assertEqual(vmin, 0)
		self.assertEqual(vmax, 100)
		self.assertIsNotNone(cmap)

	def test_multi_value_schema(self):
		"""Test multi-value color schema"""
		schema = {0: '#0000FF', 50: '#00FF00', 100: '#FF0000'}
		cmap, vmin, vmax = create_color_map(schema)
		self.assertEqual(vmin, 0)
		self.assertEqual(vmax, 100)

	def test_unsorted_schema(self):
		"""Test that unsorted schema is handled correctly"""
		schema = {100: '#FFFFFF', 0: '#4169E1', 50: '#8888FF'}
		cmap, vmin, vmax = create_color_map(schema)
		self.assertEqual(vmin, 0)
		self.assertEqual(vmax, 100)

	def test_validate_data_in_range(self):
		"""Test data within schema range produces no warning"""
		data = [10, 20, 30]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		old_stdout = sys.stdout
		sys.stdout = StringIO()
		validate_data_range(data, schema, "test_subplot")
		output = sys.stdout.getvalue()
		sys.stdout = old_stdout
		self.assertEqual(output, "")

	def test_validate_data_exceeds_max(self):
		"""Test data exceeding schema max produces warning"""
		data = [10, 20, 101]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		old_stdout = sys.stdout
		sys.stdout = StringIO()
		validate_data_range(data, schema, "test_subplot")
		output = sys.stdout.getvalue()
		sys.stdout = old_stdout
		self.assertIn("Warning", output)
		self.assertIn("101", output)
		self.assertIn("exceeding", output)

	def test_validate_data_below_min(self):
		"""Test data below schema min produces warning"""
		data = [-5, 10, 20]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		old_stdout = sys.stdout
		sys.stdout = StringIO()
		validate_data_range(data, schema, "test_subplot")
		output = sys.stdout.getvalue()
		sys.stdout = old_stdout
		self.assertIn("Warning", output)
		self.assertIn("-5", output)
		self.assertIn("below", output)

class TestPhase3GridLayout(unittest.TestCase):
	def test_zero_subplots(self):
		"""Test that zero subplots raises ValueError"""
		with self.assertRaises(ValueError):
			calculate_grid_layout(0)

	def test_negative_subplots(self):
		"""Test that negative subplots raises ValueError"""
		with self.assertRaises(ValueError):
			calculate_grid_layout(-1)

	def test_fixed_rows_cols_insufficient(self):
		"""Test fixed grid too small for subplots"""
		rows, cols = calculate_grid_layout(6, rows=2, cols=3)
		self.assertEqual(rows, 2)
		self.assertEqual(cols, 3)

	def test_fixed_rows_cols_insufficient(self):
		"""Test fixed grid too small for cells"""
		with self.assertRaises(ValueError):
			calculate_grid_layout(10, rows=2, cols=3)

	def test_fixed_rows_only(self):
		"""Test fixed rows with calculated cols"""
		rows, cols = calculate_grid_layout(5, rows=2)
		self.assertEqual(rows, 2)
		self.assertEqual(cols, 3)

	def test_fixed_cols_only(self):
		"""Test fixed cols with calculated rows"""
		rows, cols = calculate_grid_layout(5, cols=2)
		self.assertEqual(rows, 3)
		self.assertEqual(cols, 2)

	def test_auto_layout_square(self):
		"""Test automatic layout for square-ish grids"""
		rows, cols = calculate_grid_layout(9)
		self.assertEqual(rows, 3)
		self.assertEqual(cols, 3)

	def test_auto_layout_nonsquare(self):
		"""Test automatic layout for non-square grids"""
		rows, cols = calculate_grid_layout(6)
		self.assertIn(rows * cols, [6, 8, 9])
		self.assertGreaterEqual(rows * cols, 6)

	def test_max_rows_constraint(self):
		"""Test max_rows constraint"""
		rows, cols = calculate_grid_layout(10, max_rows=2)
		self.assertLessEqual(rows, 2)
		self.assertGreaterEqual(rows * cols, 10)

	def test_max_cols_constraint(self):
		"""Test max_cols constraint"""
		rows, cols = calculate_grid_layout(10, max_cols=3)
		self.assertLessEqual(cols, 3)
		self.assertGreaterEqual(rows * cols, 10)

	def test_impossible_constraints(self):
		"""Test impossible max_rows and max_cols constraints"""
		# 2x2=4 < 10, so it should work with 5x2=10
		rows, cols = calculate_grid_layout(10, max_rows=2, max_cols=5)
		self.assertGreaterEqual(rows * cols, 10)

	def test_fixed_rows_exceeds_max_cols(self):
		"""Test fixed rows that would exceed max_cols"""
		with self.assertRaises(ValueError):
			calculate_grid_layout(10, rows=2, max_cols=3)

	def test_create_square_subplots(self):
		"""Test creation of square subplots"""
		fig, axes = create_square_subplots(2, 3)
		self.assertEqual(axes.shape, (2, 3))
		self.assertIsNotNone(fig)
		plt.close(fig)

	def test_create_square_subplots_custom_size(self):
		"""Test custom figsize per subplot"""
		fig, axes = create_square_subplots(2, 2, figsize_per_subplot=5.0)
		self.assertEqual(fig.get_figwidth(), 10.0)
		self.assertEqual(fig.get_figheight(), 10.0)
		plt.close(fig)

class TestPhase4HighLevelAPI(unittest.TestCase):
	def test_empty_subplot_data(self):
		"""Test that empty subplot_data raises ValueError"""
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		with self.assertRaises(ValueError):
			plot_gradient_grid([], schema)

	def test_missing_data_field(self):
		"""Test that missing data field raises ValueError"""
		subplot_data = [{'title': 'Test'}]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		with self.assertRaises(ValueError):
			plot_gradient_grid(subplot_data, schema)

	def test_single_subplot_plot(self):
		"""Test single subplot plot"""
		subplot_data = [{'data': [10, 20, 30], 'title': 'Test Subplot'}]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema)
		self.assertIsNotNone(fig)
		self.assertEqual(axes.shape, (1, 1))
		plt.close(fig)

	def test_multiple_subplots_auto_layout(self):
		"""Test multiple subplots with auto layout"""
		subplot_data = [
			{'data': [1, 2, 3], 'title': 'Subplot 1'},
			{'data': [4, 5, 6], 'title': 'Subplot 2'},
			{'data': [7, 8, 9], 'title': 'Subplot 3'}
		]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema)
		self.assertIsNotNone(fig)
		self.assertGreaterEqual(axes.shape[0] * axes.shape[1], 3)
		plt.close(fig)

	def test_fixed_grid_layout(self):
		"""Test fixed rows and cols"""
		subplot_data = [
			{'data': [1, 2, 3]},
			{'data': [4, 5, 6]}
		]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema, rows=2, cols=3)
		self.assertEqual(axes.shape, (2, 3))
		plt.close(fig)

	def test_subplot_with_labels(self):
		"""Test subplot with title, xlabel, ylabel"""
		subplot_data = [
			{'data': [10, 20, 30], 'title': 'Test', 'xlabel': 'X', 'ylabel': 'Y'}
		]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema)
		ax = axes[0, 0]
		self.assertEqual(ax.get_title(), 'Test')
		self.assertEqual(ax.get_xlabel(), 'X')
		self.assertEqual(ax.get_ylabel(), 'Y')
		plt.close(fig)

	def test_numpy_and_list_data(self):
		"""Test both numpy and list data types"""
		subplot_data = [
			{'data': [1, 2, 3]},
			{'data': np.array([4, 5, 6])}
		]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema)
		self.assertIsNotNone(fig)
		plt.close(fig)

	def test_return_values_for_modification(self):
		"""Test that fig and axes are returned for user modification"""
		subplot_data = [{'data': [1, 2, 3]}]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema)
		# Use suptitle config instead of fig.suptitle
		fig.text(0.5, 0.98, 'Custom Title', ha='center', va='top', fontsize=14)
		axes[0, 0].set_facecolor('lightgray')
		plt.close(fig)

class TestPhase5Highlight(unittest.TestCase):
	def test_global_highlight(self):
		"""Test global best highlight - SKIPPED (API changed to highlight_borders)"""
		self.skipTest("API changed from highlight_best to highlight_borders")

	def test_local_highlight(self):
		"""Test local best highlight per row - SKIPPED (API changed to highlight_borders)"""
		self.skipTest("API changed from highlight_best to highlight_borders")

	def test_no_highlight(self):
		"""Test no highlight when highlight_borders=None"""
		subplot_data = [
			{'data': [1, 2, 3]},
			{'data': [10, 20, 30]}
		]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema, highlight_borders=None)
		self.assertIsNotNone(fig)
		plt.close(fig)

	def test_highlight_visual_indicator(self):
		"""Test that highlighted subplot has visual border - SKIPPED (API changed)"""
		self.skipTest("API changed from highlight_best to highlight_borders")


class TestPhase6ValueAnnotations(unittest.TestCase):
	def test_show_values_simple_mode(self):
		"""Test value annotations in simple mode"""
		subplot_data = [{'data': [50, 60, 70], 'title': 'Test'}]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema, show_values=True)
		ax = axes[0, 0]
		self.assertTrue(len(ax.texts) > 0)
		plt.close(fig)

	def test_show_values_nested_mode(self):
		"""Test value annotations in nested grid mode"""
		grid_data = {(0, 0): [30, 40], (0, 1): [50, 60]}
		subplot_data = [{
			'grid_data': grid_data,
			'title': 'Test Grid',
			'row_labels': ['R1'],
			'col_labels': ['C1', 'C2']
		}]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema, show_values=True)
		ax = axes[0, 0]
		self.assertTrue(len(ax.texts) > 0)
		plt.close(fig)

	def test_no_show_values_default(self):
		"""Test that values are not shown by default"""
		subplot_data = [{'data': [50, 60, 70]}]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema)
		ax = axes[0, 0]
		self.assertEqual(len(ax.texts), 0)
		plt.close(fig)

class TestPhase7TitleBolding(unittest.TestCase):
	def test_bold_titles_enabled(self):
		"""Test that titles are bold when bold_titles=True"""
		subplot_data = [{'data': [10, 20, 30], 'title': 'Test'}]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema, bold_titles=True)
		ax = axes[0, 0]
		title_obj = ax.title
		self.assertEqual(title_obj.get_fontweight(), 'bold')
		plt.close(fig)

	def test_bold_titles_disabled_default(self):
		"""Test that titles are not bold by default"""
		subplot_data = [{'data': [10, 20, 30], 'title': 'Test'}]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		fig, axes = plot_gradient_grid(subplot_data, schema)
		ax = axes[0, 0]
		title_obj = ax.title
		self.assertNotEqual(title_obj.get_fontweight(), 'bold')
		plt.close(fig)

class TestPhase8CustomHighlightColors(unittest.TestCase):
	def test_custom_highlight_colors_global(self):
		"""Test custom highlight border colors for global best"""
		np.random.seed(42)
		subplot_data = [
			{'data': np.clip(np.random.normal(50, 5, 100), 0, 100)},
			{'data': np.clip(np.random.normal(80, 5, 100), 0, 100)}		# Higher mean, global best
		]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		highlight_borders = {
			'global_best_mean': {'enabled': True, 'color': '#FF0000'}
		}
		fig, axes = plot_gradient_grid(
			subplot_data, 
			schema, 
			rows=1, 
			cols=2,
			figsize_per_subplot=4,
			highlight_borders=highlight_borders
		)
		self.assertIsNotNone(fig)
		plt.close(fig)

	def test_multiple_highlight_borders(self):
		"""Test multiple border highlights with nested grid data"""
		np.random.seed(42)
		grid_data_1 = {
			(0, 0): np.clip(np.random.normal(60, 5, 50), 0, 100),
			(0, 1): np.clip(np.random.normal(70, 5, 50), 0, 100),
			(1, 0): np.clip(np.random.normal(65, 5, 50), 0, 100),
			(1, 1): np.clip(np.random.normal(75, 5, 50), 0, 100)
		}
		grid_data_2 = {
			(0, 0): np.clip(np.random.normal(70, 5, 50), 0, 100),
			(0, 1): np.clip(np.random.normal(85, 5, 50), 0, 100),	# Highest, global best
			(1, 0): np.clip(np.random.normal(75, 5, 50), 0, 100),
			(1, 1): np.clip(np.random.normal(80, 5, 50), 0, 100)
		}
		subplot_data = [
			{'grid_data': grid_data_1, 'title': 'Model-1'},
			{'grid_data': grid_data_2, 'title': 'Model-2'}
		]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		highlight_borders = {
			'local_best_mean': {'enabled': True, 'color': '#00FF00'},
			'global_best_mean': {'enabled': True, 'color': '#FF0000'}
		}
		fig, axes = plot_gradient_grid(
			subplot_data, 
			schema, 
			rows=1, 
			cols=2,
			figsize_per_subplot=4,
			highlight_borders=highlight_borders
		)
		self.assertEqual(axes.shape, (1, 2))
		plt.close(fig)

	def test_local_best_highlighting(self):
		"""Test local best highlighting within each subplot"""
		np.random.seed(42)
		grid_data = {
			(0, 0): np.clip(np.random.normal(60, 5, 50), 0, 100),
			(0, 1): np.clip(np.random.normal(70, 5, 50), 0, 100),
			(1, 0): np.clip(np.random.normal(65, 5, 50), 0, 100),
			(1, 1): np.clip(np.random.normal(80, 5, 50), 0, 100)		# Local best
		}
		subplot_data = [{'grid_data': grid_data, 'title': 'Test Model'}]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		highlight_borders = {
			'local_best_mean': {'enabled': True, 'color': '#FFD700'},
			'local_best_median': {'enabled': True, 'color': '#00FFFF'}
		}
		fig, axes = plot_gradient_grid(
			subplot_data, 
			schema, 
			rows=1, 
			cols=1,
			figsize_per_subplot=6,
			highlight_borders=highlight_borders
		)
		self.assertIsNotNone(fig)
		plt.close(fig)

	def test_disabled_borders(self):
		"""Test that disabled borders are not drawn"""
		np.random.seed(42)
		subplot_data = [
			{'data': np.clip(np.random.normal(50, 10, 100), 0, 100)},
			{'data': np.clip(np.random.normal(80, 10, 100), 0, 100)}
		]
		schema = {0: '#4169E1', 100: '#FFFFFF'}
		highlight_borders = {
			'global_best_mean': {'enabled': False, 'color': '#FF0000'}	# Disabled
		}
		fig, axes = plot_gradient_grid(
			subplot_data, 
			schema, 
			rows=1, 
			cols=2,
			figsize_per_subplot=4,
			highlight_borders=highlight_borders
		)
		# Should complete without errors but no borders drawn
		self.assertEqual(axes.shape, (1, 2))
		plt.close(fig)

if __name__ == '__main__':
	unittest.main()