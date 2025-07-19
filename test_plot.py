#!/usr/bin/env python3
"""
Test script to verify matplotlib plotting and saving works correctly.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def test_plot():
    """Generate a simple test plot."""
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Test Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('docs/test_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Test plot saved to docs/test_plot.png")

if __name__ == "__main__":
    test_plot() 