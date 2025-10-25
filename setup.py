"""
VERSION: 2025-10-26
AUTHOR: ColorGradient Contributor(s), © 2025. All rights reserved.
LICENSE: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as file_handle:
	long_description = file_handle.read()

setup(
	name="colorgradient",
	version="1.0.0",
	author="ColorGradient Contributors", 
	author_email="",
	description="Publication-ready visualization for 1D data distributions using intuitive circular color gradients",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/PhyloBridge/ColorGradient",
	packages=find_packages(),
	classifiers=[
		"Development Status :: 5 - Production/Stable",
		"Intended Audience :: Science/Research",
		"Intended Audience :: Developers", 
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
		"Topic :: Scientific/Engineering :: Visualization",
		"Topic :: Software Development :: Libraries :: Python Modules",
		"License :: Free for non-commercial use",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8", 
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.11",
		"Programming Language :: Python :: 3.12",
		"Programming Language :: Python :: 3.13",
		"Programming Language :: Python :: 3.14",
		"Operating System :: OS Independent",
	],
	python_requires=">=3.7",
	install_requires=[
		"numpy>=1.19.0",
		"matplotlib>=3.3.0",
	],
	keywords=[
		"hyperparameter-optimization", "machine-learning", "visualization", 
		"gradient", "heatmap", "grid-search", "cross-validation",
		"deep-learning", "neural-networks", "model-comparison",
		"performance-visualization", "scientific-plotting", "publication",
		"research-tools", "data-science", "pytorch", "tensorflow",
		"hyperparameter-tuning", "model-selection", "experiment-tracking",
		"automl", "neural-architecture-search", "bayesian-optimization",
		"colorbar", "legend", "circular-gradient", "distribution-visualization"
	],
	project_urls={
		"Bug Reports": "https://github.com/PhyloBridge/ColorGradient/issues",
		"Source": "https://github.com/PhyloBridge/ColorGradient", 
		"Documentation": "https://github.com/PhyloBridge/ColorGradient#readme",
		"Examples": "https://github.com/PhyloBridge/ColorGradient/tree/main/examples"
	}
)