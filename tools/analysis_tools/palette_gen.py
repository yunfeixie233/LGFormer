def generate_palette(palette_part, total_colors=159, step=15, min_difference=30):
    """
    Generate a palette of unique colors.

    :param palette_part: List of initial colors in the palette.
    :param total_colors: Total number of colors to generate.
    :param step: Step size for changing RGB values.
    :param min_difference: Minimum RGB difference to consider a color unique.
    :return: List of colors in the palette.
    """
    from itertools import product

    def is_color_unique(new_color, palette):
        """ Check if the new color is sufficiently different from all colors in the palette. """
        return all(sum(abs(c1 - c2) for c1, c2 in zip(new_color, color)) >= min_difference for color in palette)

    palette = palette_part.copy()

    # Create all possible RGB adjustments within the step size
    adjustments = list(product([-step, 0, step], repeat=3))

    while len(palette) < total_colors:
        for color in palette.copy():
            for adjustment in adjustments:
                new_color = [max(0, min(255, c + a)) for c, a in zip(color, adjustment)]
                if new_color not in palette and is_color_unique(new_color, palette):
                    palette.append(new_color)
                    break
            if len(palette) >= total_colors:
                break

    return palette

# Given partial palette
palette_part = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                [0, 0, 0]]

# Generate the full palette
full_palette = generate_palette(palette_part)
print(full_palette)  # Display the length and first 10 colors of the palette
