import re
def extract_iou_from_table_clean(table_text):
    # Define a regex pattern to extract class names and their IoU values
    pattern = r'\|\s*(.*?)\s*\|\s*([\d\.]+)\s*\|'
    
    # Extract matches using regex
    matches = re.findall(pattern, table_text)

    # Convert matches to a dictionary and ensure class names are clean
    iou_values = {}
    for match in matches:
        class_name = match[0].strip().replace('|', '').strip()  # Cleaning up class name
        iou = float(match[1])
        iou_values[class_name] = iou
    
    return iou_values
def average_iou(tables):
    """Compute the average IoU values from a list of tables."""
    sum_iou_values = {}
    num_tables = len(tables)
    
    # Sum IoU values from all tables
    for table in tables:
        iou_values = extract_iou_from_table_clean(table)
        for class_name, iou in iou_values.items():
            sum_iou_values[class_name] = sum_iou_values.get(class_name, 0) + iou
    
    # Calculate average IoU
    avg_iou_values = {class_name: total_iou / num_tables for class_name, total_iou in sum_iou_values.items()}
    
    return avg_iou_values

def compare_average_iou_values(tables_A, tables_B, k=5):
    """Compare average IoU values from two sets of tables and return top k differences."""
    # Compute average IoU for both sets of tables
    avg_iou_values_A = average_iou(tables_A)
    avg_iou_values_B = average_iou(tables_B)
    
    # Compute differences between the average IoU values
    differences = {}
    for class_name in avg_iou_values_A:
        if class_name in avg_iou_values_B:
            diff_A_B = avg_iou_values_A[class_name] - avg_iou_values_B[class_name]
            differences[class_name] = diff_A_B

    # Sort and return top k differences
    sorted_differences = sorted(differences.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_differences[:k]

# Sample tables for testing
table_A1 = """
+---------------------+-------+-------+
|        Class        |  IoU  |  Acc  |
+---------------------+-------+-------+
|         wall        |  75.3 | 88.16 |
|       building      | 81.56 | 92.21 |
|         sky         | 94.18 | 97.41 |
|        floor        | 79.75 |  91.0 |
|         tree        | 73.66 | 87.07 |
|       ceiling       | 82.85 | 92.35 |
|         road        |  83.3 | 90.27 |
|         bed         | 85.64 | 94.56 |
|      windowpane     | 59.51 | 77.77 |
|        grass        | 65.53 |  81.8 |
|       cabinet       | 60.04 | 74.16 |
|       sidewalk      | 64.31 | 80.45 |
|        person       | 77.81 | 91.58 |
|        earth        | 34.48 | 46.92 |
|         door        |  45.1 | 57.44 |
|        table        | 55.08 | 74.29 |
|       mountain      | 57.73 | 73.04 |
|        plant        | 48.66 | 61.19 |
|       curtain       | 70.45 | 81.19 |
|        chair        | 53.05 | 70.16 |
|         car         | 81.86 | 90.87 |
|        water        | 55.43 | 68.01 |
|       painting      |  70.1 |  84.2 |
|         sofa        |  59.5 | 74.95 |
|        shelf        | 43.36 | 63.25 |
|        house        |  39.8 | 48.25 |
|         sea         | 65.42 | 88.55 |
|        mirror       | 65.25 | 74.59 |
|         rug         | 62.34 |  68.1 |
|        field        |  29.6 | 45.88 |
|       armchair      | 38.86 | 54.39 |
|         seat        | 62.33 | 79.95 |
|        fence        | 40.49 | 54.74 |
|         desk        | 46.83 | 65.19 |
|         rock        | 40.94 | 63.98 |
|       wardrobe      | 52.25 | 63.51 |
|         lamp        | 59.13 | 71.46 |
|       bathtub       | 72.36 | 77.57 |
|       railing       | 31.44 | 45.55 |
|       cushion       | 53.41 | 69.75 |
|         base        | 17.12 | 24.53 |
|         box         | 20.19 | 27.12 |
|        column       | 41.66 | 48.17 |
|      signboard      | 35.24 | 48.71 |
|   chest of drawers  | 39.21 | 49.98 |
|       counter       | 23.85 | 30.26 |
|         sand        | 42.18 | 63.96 |
|         sink        | 69.85 | 77.38 |
|      skyscraper     | 69.31 |  82.8 |
|      fireplace      | 74.12 | 85.02 |
|     refrigerator    | 69.56 | 77.25 |
|      grandstand     | 40.99 | 61.87 |
|         path        | 21.68 |  30.6 |
|        stairs       | 22.04 | 28.56 |
|        runway       | 74.99 | 95.53 |
|         case        | 44.05 | 54.41 |
|      pool table     |  93.1 | 96.24 |
|        pillow       | 52.91 | 67.11 |
|     screen door     | 64.81 | 70.16 |
|       stairway      | 32.95 | 42.14 |
|        river        |  11.7 |  24.4 |
|        bridge       | 57.15 |  75.9 |
|       bookcase      | 34.82 | 49.24 |
|        blind        | 37.29 | 39.81 |
|     coffee table    | 58.99 | 76.67 |
|        toilet       | 83.42 | 88.98 |
|        flower       | 37.78 | 57.32 |
|         book        | 44.37 | 61.58 |
|         hill        |  6.93 | 10.48 |
|        bench        | 42.07 | 47.51 |
|      countertop     | 52.23 | 66.95 |
|        stove        | 74.09 | 80.46 |
|         palm        | 50.12 | 69.25 |
|    kitchen island   |  38.1 |  65.3 |
|       computer      | 59.71 | 66.83 |
|     swivel chair    | 36.85 | 47.09 |
|         boat        | 68.91 | 80.29 |
|         bar         | 50.82 | 56.59 |
|    arcade machine   |  68.6 | 71.23 |
|        hovel        | 21.53 |  23.3 |
|         bus         | 83.36 | 89.61 |
|        towel        | 58.81 | 68.37 |
|        light        | 49.41 | 59.69 |
|        truck        | 22.71 | 31.55 |
|        tower        | 26.35 | 33.96 |
|      chandelier     | 63.16 |  78.8 |
|        awning       | 28.28 | 33.84 |
|     streetlight     | 20.72 | 25.85 |
|        booth        | 47.72 | 49.97 |
| television receiver | 67.19 | 77.63 |
|       airplane      |  52.7 |  63.1 |
|      dirt track     |  0.02 |  0.04 |
|       apparel       | 30.93 | 43.01 |
|         pole        | 19.54 | 25.75 |
|         land        |  4.94 |  8.68 |
|      bannister      |  8.16 | 11.04 |
|      escalator      | 24.43 | 25.67 |
|       ottoman       | 39.06 | 54.64 |
|        bottle       | 36.94 | 48.39 |
|        buffet       | 28.85 | 31.83 |
|        poster       | 29.96 | 37.13 |
|        stage        | 14.32 | 22.21 |
|         van         | 43.43 | 60.39 |
|         ship        | 59.93 | 63.39 |
|       fountain      | 29.41 | 31.79 |
|    conveyer belt    | 78.61 | 91.22 |
|        canopy       | 14.29 | 15.67 |
|        washer       | 67.25 | 69.77 |
|      plaything      | 20.52 | 28.44 |
|    swimming pool    | 54.12 | 83.74 |
|        stool        | 36.12 | 46.82 |
|        barrel       |  3.87 |  7.01 |
|        basket       | 33.58 | 42.44 |
|      waterfall      | 57.15 | 68.64 |
|         tent        | 87.73 | 98.03 |
|         bag         | 11.77 | 15.34 |
|       minibike      | 54.85 | 67.92 |
|        cradle       | 78.33 | 95.02 |
|         oven        |  29.6 | 49.47 |
|         ball        | 42.13 | 49.11 |
|         food        | 47.59 | 52.54 |
|         step        |  9.24 | 10.25 |
|         tank        | 46.66 | 51.19 |
|      trade name     | 19.54 | 22.81 |
|      microwave      | 65.45 | 72.74 |
|         pot         | 33.71 | 39.17 |
|        animal       | 56.52 | 59.85 |
|       bicycle       | 46.14 | 70.93 |
|         lake        | 58.15 | 63.43 |
|      dishwasher     | 61.38 | 69.92 |
|        screen       | 66.56 | 79.09 |
|       blanket       |  7.57 |  9.11 |
|      sculpture      | 49.95 | 59.85 |
|         hood        | 59.05 | 63.68 |
|        sconce       |  43.6 |  52.2 |
|         vase        | 28.13 | 43.22 |
|    traffic light    | 31.48 | 45.45 |
|         tray        |  7.33 |  13.9 |
|        ashcan       | 37.66 | 46.99 |
|         fan         | 56.76 | 72.05 |
|         pier        | 29.17 |  53.2 |
|      crt screen     | 16.84 | 30.39 |
|        plate        | 51.83 | 67.63 |
|       monitor       | 55.58 | 67.83 |
|    bulletin board   | 35.59 | 42.58 |
|        shower       |  1.93 |  3.57 |
|       radiator      | 61.12 | 68.25 |
|        glass        |  6.77 |  7.01 |
|        clock        |  26.5 | 31.69 |
|         flag        | 40.49 | 43.74 |
+---------------------+-------+-------+
"""
table_A2 = """
+---------------------+-------+-------+
|        Class        |  IoU  |  Acc  |
+---------------------+-------+-------+
|         wall        | 75.08 | 87.99 |
|       building      |  81.7 | 92.04 |
|         sky         | 94.19 | 97.29 |
|        floor        | 79.76 | 90.57 |
|         tree        | 73.36 | 87.99 |
|       ceiling       | 83.19 | 92.23 |
|         road        | 83.32 | 90.13 |
|         bed         | 85.59 | 93.84 |
|      windowpane     | 59.74 | 76.98 |
|        grass        |  65.6 | 81.39 |
|       cabinet       | 60.27 | 75.07 |
|       sidewalk      | 64.23 | 80.04 |
|        person       | 77.77 | 92.11 |
|        earth        | 35.17 | 48.41 |
|         door        | 45.41 | 58.63 |
|        table        | 55.18 | 73.44 |
|       mountain      | 57.72 | 73.08 |
|        plant        | 47.77 | 59.58 |
|       curtain       | 70.56 |  82.2 |
|        chair        | 52.58 | 68.17 |
|         car         |  81.6 | 91.17 |
|        water        | 56.29 | 68.61 |
|       painting      | 68.89 | 85.13 |
|         sofa        | 59.18 | 75.49 |
|        shelf        | 43.64 | 62.14 |
|        house        | 47.05 | 57.29 |
|         sea         | 66.62 | 90.22 |
|        mirror       | 64.71 | 73.23 |
|         rug         | 60.36 | 65.52 |
|        field        | 29.72 | 46.54 |
|       armchair      | 38.38 | 57.07 |
|         seat        | 61.42 | 79.13 |
|        fence        | 40.32 | 55.01 |
|         desk        | 46.35 | 63.19 |
|         rock        | 40.32 | 65.21 |
|       wardrobe      | 51.72 | 63.64 |
|         lamp        | 59.43 | 72.82 |
|       bathtub       | 73.05 | 78.95 |
|       railing       | 32.25 | 47.22 |
|       cushion       | 52.95 | 69.33 |
|         base        | 19.86 | 30.38 |
|         box         | 20.12 | 26.57 |
|        column       | 41.16 |  49.0 |
|      signboard      | 35.13 | 48.88 |
|   chest of drawers  | 38.12 | 46.74 |
|       counter       |  25.1 | 33.14 |
|         sand        |  41.0 | 60.93 |
|         sink        | 69.66 | 76.72 |
|      skyscraper     | 66.65 | 76.76 |
|      fireplace      | 73.89 | 86.62 |
|     refrigerator    | 68.58 | 76.48 |
|      grandstand     |  39.8 | 61.71 |
|         path        | 22.51 | 31.64 |
|        stairs       |  24.0 | 30.76 |
|        runway       | 74.79 | 95.02 |
|         case        | 43.54 | 52.74 |
|      pool table     | 92.96 | 96.42 |
|        pillow       | 53.69 | 69.29 |
|     screen door     | 66.06 | 72.77 |
|       stairway      | 33.54 | 41.85 |
|        river        | 12.57 | 25.54 |
|        bridge       | 59.18 | 74.18 |
|       bookcase      | 34.94 | 50.96 |
|        blind        | 38.88 | 41.96 |
|     coffee table    | 58.04 | 78.86 |
|        toilet       | 83.09 | 88.69 |
|        flower       | 37.76 | 55.57 |
|         book        | 45.25 | 61.99 |
|         hill        |  4.55 |  6.29 |
|        bench        | 45.73 | 52.24 |
|      countertop     | 51.51 | 67.89 |
|        stove        | 73.77 | 81.59 |
|         palm        | 49.72 | 69.16 |
|    kitchen island   | 40.33 | 62.93 |
|       computer      |  59.1 | 67.81 |
|     swivel chair    | 37.95 | 47.99 |
|         boat        |  70.5 |  80.2 |
|         bar         | 51.58 | 57.11 |
|    arcade machine   | 66.86 | 69.58 |
|        hovel        | 23.44 | 25.59 |
|         bus         | 82.35 | 90.16 |
|        towel        | 61.23 | 70.04 |
|        light        | 48.73 | 56.42 |
|        truck        | 21.01 |  29.2 |
|        tower        | 26.82 | 36.14 |
|      chandelier     | 62.89 | 79.96 |
|        awning       | 30.02 | 35.56 |
|     streetlight     | 21.28 | 26.57 |
|        booth        | 50.16 |  51.6 |
| television receiver | 67.12 | 78.08 |
|       airplane      | 52.87 | 61.78 |
|      dirt track     |  0.01 |  0.02 |
|       apparel       | 32.26 | 42.87 |
|         pole        | 19.66 | 26.16 |
|         land        |  4.91 |  8.42 |
|      bannister      |  8.36 | 11.36 |
|      escalator      | 23.61 | 25.07 |
|       ottoman       | 39.18 | 56.85 |
|        bottle       | 36.66 | 50.39 |
|        buffet       | 31.29 | 34.92 |
|        poster       | 26.33 | 31.46 |
|        stage        | 12.58 | 18.78 |
|         van         | 41.84 | 57.33 |
|         ship        | 50.64 | 53.91 |
|       fountain      | 27.25 | 29.26 |
|    conveyer belt    | 78.85 | 91.36 |
|        canopy       |  16.7 | 18.66 |
|        washer       | 66.89 | 69.18 |
|      plaything      | 19.76 | 27.49 |
|    swimming pool    | 53.56 | 84.37 |
|        stool        | 36.14 | 47.89 |
|        barrel       |  4.01 |  6.69 |
|        basket       | 32.04 | 40.62 |
|      waterfall      | 63.33 | 76.86 |
|         tent        | 86.43 |  98.2 |
|         bag         | 12.15 | 15.56 |
|       minibike      | 48.37 | 58.79 |
|        cradle       | 76.07 | 93.47 |
|         oven        | 26.57 | 40.83 |
|         ball        | 35.25 | 39.78 |
|         food        | 48.54 | 55.06 |
|         step        |  9.49 | 10.68 |
|         tank        |  47.0 |  50.5 |
|      trade name     | 17.16 | 19.65 |
|      microwave      | 67.83 | 76.33 |
|         pot         | 34.38 | 40.92 |
|        animal       | 54.61 | 57.79 |
|       bicycle       | 46.73 | 71.34 |
|         lake        | 58.82 | 63.18 |
|      dishwasher     | 57.91 | 69.95 |
|        screen       | 65.13 | 78.68 |
|       blanket       |  8.63 | 10.06 |
|      sculpture      | 47.62 | 60.62 |
|         hood        | 60.25 | 64.22 |
|        sconce       | 41.92 | 50.29 |
|         vase        | 28.44 | 44.24 |
|    traffic light    | 31.53 | 44.07 |
|         tray        |  7.54 | 13.97 |
|        ashcan       | 37.44 | 46.81 |
|         fan         | 57.16 | 70.78 |
|         pier        | 29.36 | 54.15 |
|      crt screen     | 17.81 | 29.97 |
|        plate        | 52.54 | 68.42 |
|       monitor       | 58.52 | 70.98 |
|    bulletin board   | 31.85 | 40.74 |
|        shower       |  2.75 |  5.18 |
|       radiator      | 61.71 |  69.8 |
|        glass        |  7.85 |  8.17 |
|        clock        | 26.53 | 32.28 |
|         flag        | 40.47 | 43.89 |
+---------------------+-------+-------+
"""  # For simplicity in this example
table_A3 = """
+---------------------+-------+-------+
|        Class        |  IoU  |  Acc  |
+---------------------+-------+-------+
|         wall        | 75.19 | 87.48 |
|       building      | 81.66 | 91.55 |
|         sky         | 94.16 | 97.44 |
|        floor        | 79.85 | 90.92 |
|         tree        | 73.67 | 88.14 |
|       ceiling       | 82.82 | 92.25 |
|         road        | 83.57 | 90.45 |
|         bed         | 85.45 | 94.17 |
|      windowpane     | 59.88 | 77.93 |
|        grass        | 65.35 |  82.2 |
|       cabinet       | 60.06 | 74.47 |
|       sidewalk      | 64.35 | 79.47 |
|        person       | 77.91 | 91.27 |
|        earth        | 35.49 | 48.58 |
|         door        | 45.65 | 60.87 |
|        table        | 54.28 |  74.0 |
|       mountain      | 58.27 | 71.92 |
|        plant        |  47.0 | 58.87 |
|       curtain       | 70.93 | 82.92 |
|        chair        | 52.36 |  67.8 |
|         car         | 81.61 | 90.56 |
|        water        | 56.16 |  68.3 |
|       painting      | 69.54 | 85.18 |
|         sofa        | 58.76 | 74.65 |
|        shelf        | 43.96 | 60.89 |
|        house        | 48.71 |  62.7 |
|         sea         | 65.44 | 87.77 |
|        mirror       | 64.52 | 73.76 |
|         rug         | 62.36 | 68.54 |
|        field        | 29.32 | 44.85 |
|       armchair      | 38.77 | 57.68 |
|         seat        | 60.83 |  79.9 |
|        fence        | 40.29 | 54.44 |
|         desk        | 47.06 | 65.66 |
|         rock        | 40.46 | 65.81 |
|       wardrobe      | 52.08 | 65.23 |
|         lamp        | 58.89 | 71.49 |
|       bathtub       | 72.67 | 78.74 |
|       railing       | 31.75 |  46.0 |
|       cushion       | 52.92 | 67.82 |
|         base        | 20.65 | 30.21 |
|         box         | 20.28 | 25.82 |
|        column       | 42.63 | 50.14 |
|      signboard      |  34.6 |  46.3 |
|   chest of drawers  | 39.84 | 52.56 |
|       counter       | 23.43 | 29.02 |
|         sand        |  42.1 | 67.05 |
|         sink        | 69.74 | 78.05 |
|      skyscraper     |  68.2 | 84.13 |
|      fireplace      | 73.57 | 85.28 |
|     refrigerator    |  69.3 | 76.96 |
|      grandstand     | 41.19 | 60.78 |
|         path        | 22.51 | 31.58 |
|        stairs       | 22.88 | 29.96 |
|        runway       | 74.32 | 94.84 |
|         case        |  43.2 | 55.46 |
|      pool table     | 92.98 |  96.3 |
|        pillow       | 52.57 | 65.78 |
|     screen door     | 62.54 | 68.79 |
|       stairway      | 34.26 | 42.42 |
|        river        | 13.07 | 26.76 |
|        bridge       | 56.44 | 73.61 |
|       bookcase      | 34.14 | 50.57 |
|        blind        | 39.95 |  43.3 |
|     coffee table    | 58.23 | 79.22 |
|        toilet       | 83.19 | 89.23 |
|        flower       | 36.87 | 57.49 |
|         book        | 45.51 | 63.76 |
|         hill        |  4.92 |  7.06 |
|        bench        | 43.88 | 49.91 |
|      countertop     | 51.43 | 64.65 |
|        stove        |  73.2 | 81.17 |
|         palm        | 49.48 |  66.0 |
|    kitchen island   | 37.84 | 64.49 |
|       computer      | 61.54 | 70.73 |
|     swivel chair    | 38.39 | 50.76 |
|         boat        | 65.65 | 79.59 |
|         bar         | 48.39 | 54.75 |
|    arcade machine   | 71.76 | 75.11 |
|        hovel        | 30.37 | 34.24 |
|         bus         | 82.62 | 90.13 |
|        towel        |  60.1 | 68.05 |
|        light        | 49.24 | 58.31 |
|        truck        | 22.73 | 32.23 |
|        tower        | 30.98 | 42.68 |
|      chandelier     | 62.25 | 79.93 |
|        awning       | 29.13 | 34.49 |
|     streetlight     | 20.52 | 25.26 |
|        booth        | 51.77 |  54.1 |
| television receiver |  66.3 | 77.34 |
|       airplane      | 53.48 | 62.72 |
|      dirt track     |  0.0  |  0.0  |
|       apparel       | 31.29 | 42.35 |
|         pole        | 19.58 | 25.31 |
|         land        |  4.92 |  7.99 |
|      bannister      |  8.11 | 11.03 |
|      escalator      | 25.32 | 26.76 |
|       ottoman       | 40.68 |  57.2 |
|        bottle       | 37.09 |  49.3 |
|        buffet       | 37.02 | 42.12 |
|        poster       | 24.48 | 29.07 |
|        stage        | 12.09 | 20.78 |
|         van         |  42.4 | 60.08 |
|         ship        | 60.16 | 63.87 |
|       fountain      | 28.62 |  30.6 |
|    conveyer belt    |  80.3 | 91.04 |
|        canopy       | 20.76 | 23.07 |
|        washer       | 67.63 | 70.18 |
|      plaything      | 19.57 | 27.22 |
|    swimming pool    | 54.25 | 83.38 |
|        stool        |  35.9 | 48.06 |
|        barrel       |  1.26 |  1.97 |
|        basket       | 32.62 | 42.14 |
|      waterfall      | 64.02 | 75.99 |
|         tent        | 89.66 | 98.17 |
|         bag         | 11.29 | 14.47 |
|       minibike      | 56.01 | 68.41 |
|        cradle       | 77.55 | 96.15 |
|         oven        |  26.4 | 44.77 |
|         ball        | 44.95 | 53.08 |
|         food        | 46.65 | 51.99 |
|         step        |  9.5  | 10.73 |
|         tank        | 48.93 | 53.06 |
|      trade name     | 20.26 | 23.94 |
|      microwave      | 65.56 | 72.37 |
|         pot         | 34.37 | 39.98 |
|        animal       | 54.74 |  57.7 |
|       bicycle       | 47.45 | 71.91 |
|         lake        | 57.96 | 63.34 |
|      dishwasher     | 59.73 | 70.23 |
|        screen       | 64.78 | 78.94 |
|       blanket       |  7.99 |  9.72 |
|      sculpture      | 48.88 | 59.46 |
|         hood        | 59.63 |  64.2 |
|        sconce       |  43.0 |  51.5 |
|         vase        | 28.58 | 44.39 |
|    traffic light    | 31.09 | 43.15 |
|         tray        |  7.7  | 14.29 |
|        ashcan       | 37.47 | 48.08 |
|         fan         | 56.35 | 71.09 |
|         pier        | 28.24 | 56.96 |
|      crt screen     | 17.57 | 31.23 |
|        plate        | 51.77 | 68.95 |
|       monitor       | 53.66 | 64.27 |
|    bulletin board   | 32.91 |  40.7 |
|        shower       |  2.43 |  4.36 |
|       radiator      | 61.93 | 71.33 |
|        glass        |  7.04 |  7.24 |
|        clock        | 25.67 | 31.03 |
|         flag        | 40.06 | 43.25 |
+---------------------+-------+-------+
"""

table_B1 = """
+---------------------+-------+-------+
|        Class        |  IoU  |  Acc  |
+---------------------+-------+-------+
|         wall        | 74.76 | 87.58 |
|       building      | 80.15 | 91.29 |
|         sky         | 93.81 | 96.93 |
|        floor        |  79.5 | 90.57 |
|         tree        | 72.31 | 88.12 |
|       ceiling       |  83.2 | 92.79 |
|         road        | 81.19 |  88.5 |
|         bed         | 84.56 | 93.29 |
|      windowpane     | 59.48 | 75.79 |
|        grass        | 67.75 | 86.18 |
|       cabinet       | 56.56 | 70.38 |
|       sidewalk      | 64.11 | 80.49 |
|        person       | 77.82 | 90.39 |
|        earth        | 36.19 | 48.38 |
|         door        | 41.68 | 55.86 |
|        table        | 54.19 | 71.71 |
|       mountain      | 55.05 | 68.09 |
|        plant        | 47.89 | 59.66 |
|       curtain       | 69.78 | 82.41 |
|        chair        | 51.78 | 66.15 |
|         car         | 80.82 | 88.99 |
|        water        | 48.05 | 63.39 |
|       painting      | 70.46 | 86.53 |
|         sofa        | 59.65 | 76.77 |
|        shelf        | 39.54 | 56.33 |
|        house        | 38.34 | 47.53 |
|         sea         | 50.37 | 75.25 |
|        mirror       | 63.24 | 71.11 |
|         rug         | 61.37 | 66.95 |
|        field        | 31.54 | 43.29 |
|       armchair      | 37.48 | 54.45 |
|         seat        | 53.82 |  72.2 |
|        fence        |  42.3 | 57.14 |
|         desk        | 45.11 | 68.16 |
|         rock        | 45.17 | 72.37 |
|       wardrobe      | 42.16 | 57.88 |
|         lamp        | 58.59 | 70.56 |
|       bathtub       | 73.09 | 78.36 |
|       railing       | 31.13 | 41.76 |
|       cushion       | 53.69 | 69.89 |
|         base        |  19.5 | 28.11 |
|         box         | 22.71 | 31.22 |
|        column       | 43.56 | 54.89 |
|      signboard      | 33.27 | 45.41 |
|   chest of drawers  | 35.96 | 55.39 |
|       counter       | 29.87 | 36.42 |
|         sand        | 54.74 | 70.96 |
|         sink        | 66.88 | 77.12 |
|      skyscraper     | 63.83 | 75.71 |
|      fireplace      | 69.86 | 81.05 |
|     refrigerator    | 70.72 | 81.07 |
|      grandstand     | 36.81 | 68.84 |
|         path        | 25.93 | 38.34 |
|        stairs       | 28.68 | 36.93 |
|        runway       | 70.13 | 91.14 |
|         case        | 46.16 | 57.26 |
|      pool table     | 91.15 | 95.55 |
|        pillow       | 54.38 | 64.59 |
|     screen door     |  52.5 | 57.61 |
|       stairway      |  28.7 | 38.78 |
|        river        | 12.73 | 24.49 |
|        bridge       | 60.62 | 72.19 |
|       bookcase      | 37.25 | 51.41 |
|        blind        | 38.42 | 41.58 |
|     coffee table    | 57.67 | 75.76 |
|        toilet       | 82.37 | 88.74 |
|        flower       | 34.39 |  54.1 |
|         book        | 45.96 |  65.4 |
|         hill        | 13.16 | 18.37 |
|        bench        | 39.47 | 46.34 |
|      countertop     |  50.6 | 62.81 |
|        stove        | 73.38 | 79.58 |
|         palm        |  48.8 | 63.71 |
|    kitchen island   |  36.1 |  64.7 |
|       computer      | 57.69 | 65.66 |
|     swivel chair    | 44.56 | 63.94 |
|         boat        | 46.44 | 53.63 |
|         bar         | 30.56 |  37.9 |
|    arcade machine   | 61.95 | 64.35 |
|        hovel        | 13.69 | 15.33 |
|         bus         | 88.51 | 93.79 |
|        towel        | 55.07 | 66.43 |
|        light        | 49.83 | 57.56 |
|        truck        | 19.19 |  33.0 |
|        tower        | 23.24 | 38.38 |
|      chandelier     | 61.66 | 78.44 |
|        awning       | 26.23 | 31.18 |
|     streetlight     | 20.76 | 24.78 |
|        booth        | 58.17 | 62.79 |
| television receiver | 64.76 |  77.2 |
|       airplane      | 52.62 | 63.97 |
|      dirt track     |  4.28 | 18.83 |
|       apparel       | 28.69 | 39.85 |
|         pole        | 22.69 | 31.45 |
|         land        |  3.81 |  5.26 |
|      bannister      |  9.47 | 13.46 |
|      escalator      | 28.98 |  35.0 |
|       ottoman       | 43.18 | 59.81 |
|        bottle       | 33.31 | 49.33 |
|        buffet       | 43.45 | 47.31 |
|        poster       | 31.45 | 35.91 |
|        stage        | 11.31 | 16.93 |
|         van         | 43.15 | 56.95 |
|         ship        | 61.97 | 85.02 |
|       fountain      | 26.52 | 29.37 |
|    conveyer belt    | 83.61 |  90.8 |
|        canopy       | 14.08 | 15.46 |
|        washer       | 69.57 | 72.77 |
|      plaything      | 21.42 | 30.04 |
|    swimming pool    | 57.19 |  80.1 |
|        stool        | 33.59 | 47.13 |
|        barrel       | 39.31 | 45.12 |
|        basket       | 32.48 | 39.92 |
|      waterfall      | 40.89 | 59.13 |
|         tent        | 91.81 | 98.03 |
|         bag         | 10.86 | 14.17 |
|       minibike      | 58.81 | 73.54 |
|        cradle       | 79.35 | 95.91 |
|         oven        |  28.9 | 36.21 |
|         ball        | 46.75 | 61.82 |
|         food        | 48.66 | 57.03 |
|         step        |  7.78 |  9.36 |
|         tank        | 42.05 | 52.96 |
|      trade name     | 21.97 | 25.23 |
|      microwave      | 70.81 | 78.97 |
|         pot         | 35.63 | 41.73 |
|        animal       | 51.19 | 53.63 |
|       bicycle       | 53.44 | 73.39 |
|         lake        | 58.07 | 65.63 |
|      dishwasher     | 57.15 | 70.05 |
|        screen       | 73.76 |  86.1 |
|       blanket       |  9.16 |  10.6 |
|      sculpture      | 60.19 |  75.0 |
|         hood        | 62.83 | 68.04 |
|        sconce       | 40.83 | 51.75 |
|         vase        | 28.14 | 43.44 |
|    traffic light    |  35.9 | 47.55 |
|         tray        |  9.48 | 13.74 |
|        ashcan       | 36.65 | 50.04 |
|         fan         | 52.28 | 62.62 |
|         pier        | 51.87 | 86.98 |
|      crt screen     | 12.73 | 27.31 |
|        plate        | 47.36 | 59.56 |
|       monitor       | 28.64 | 33.08 |
|    bulletin board   | 39.91 |  59.3 |
|        shower       |  2.56 |  14.2 |
|       radiator      | 55.57 | 61.66 |
|        glass        |  7.17 |  7.54 |
|        clock        | 33.57 | 38.19 |
|         flag        | 31.84 | 35.23 |
+---------------------+-------+-------+
"""
table_B2 = """
+---------------------+-------+-------+
|        Class        |  IoU  |  Acc  |
+---------------------+-------+-------+
|         wall        | 74.98 | 87.72 |
|       building      |  79.8 | 91.41 |
|         sky         | 93.75 | 96.86 |
|        floor        | 79.66 | 91.23 |
|         tree        | 72.26 | 88.25 |
|       ceiling       | 83.16 | 92.64 |
|         road        | 80.82 | 88.13 |
|         bed         | 84.65 | 93.82 |
|      windowpane     | 58.86 | 73.43 |
|        grass        | 66.88 |  82.4 |
|       cabinet       | 57.37 | 71.25 |
|       sidewalk      | 64.15 | 81.31 |
|        person       | 77.46 | 91.11 |
|        earth        | 35.24 | 50.08 |
|         door        | 41.29 | 56.51 |
|        table        | 54.34 | 74.46 |
|       mountain      |  55.9 | 69.42 |
|        plant        | 47.36 | 58.56 |
|       curtain       | 70.65 | 82.05 |
|        chair        | 51.69 | 65.23 |
|         car         | 80.69 | 89.15 |
|        water        | 48.91 | 64.15 |
|       painting      | 70.84 | 87.06 |
|         sofa        | 58.96 | 74.96 |
|        shelf        | 39.01 | 55.84 |
|        house        | 30.59 | 35.31 |
|         sea         | 47.53 | 69.74 |
|        mirror       | 62.76 | 73.68 |
|         rug         | 61.13 | 66.38 |
|        field        | 29.28 | 44.67 |
|       armchair      | 37.15 | 55.09 |
|         seat        | 53.64 | 73.47 |
|        fence        | 40.74 | 55.11 |
|         desk        | 45.75 | 64.23 |
|         rock        | 44.56 | 68.83 |
|       wardrobe      | 44.27 | 55.06 |
|         lamp        | 58.37 | 70.95 |
|       bathtub       | 72.15 | 77.57 |
|       railing       | 31.08 | 43.73 |
|       cushion       | 52.82 | 70.42 |
|         base        | 19.15 | 31.45 |
|         box         | 21.21 | 29.03 |
|        column       | 41.52 | 50.96 |
|      signboard      | 33.57 | 45.97 |
|   chest of drawers  | 33.76 | 53.33 |
|       counter       | 25.48 | 30.44 |
|         sand        | 52.19 | 69.34 |
|         sink        |  66.5 | 77.71 |
|      skyscraper     | 63.69 | 77.92 |
|      fireplace      | 73.14 | 85.76 |
|     refrigerator    | 69.93 | 77.26 |
|      grandstand     |  33.5 | 66.67 |
|         path        | 19.53 | 26.59 |
|        stairs       | 29.53 |  37.0 |
|        runway       | 71.65 |  87.8 |
|         case        |  45.8 |  56.8 |
|      pool table     | 91.36 | 95.32 |
|        pillow       | 54.63 | 65.53 |
|     screen door     | 56.07 | 59.58 |
|       stairway      | 29.46 | 39.87 |
|        river        | 15.43 | 29.29 |
|        bridge       |  52.4 | 60.54 |
|       bookcase      | 37.05 | 49.29 |
|        blind        |  40.8 | 47.77 |
|     coffee table    | 58.23 | 75.37 |
|        toilet       | 82.11 | 87.86 |
|        flower       | 33.83 | 52.32 |
|         book        | 45.55 | 64.15 |
|         hill        | 11.88 | 16.75 |
|        bench        | 39.47 | 44.65 |
|      countertop     | 49.84 | 61.05 |
|        stove        | 73.04 | 80.51 |
|         palm        | 49.35 | 66.96 |
|    kitchen island   | 36.97 | 64.22 |
|       computer      | 58.69 | 68.06 |
|     swivel chair    | 43.35 | 57.25 |
|         boat        | 48.63 |  56.3 |
|         bar         | 31.63 | 40.74 |
|    arcade machine   | 68.18 | 70.93 |
|        hovel        | 14.77 | 16.18 |
|         bus         |  87.9 | 92.63 |
|        towel        | 54.71 | 64.28 |
|        light        | 48.46 | 54.18 |
|        truck        | 20.55 | 29.56 |
|        tower        |  20.5 | 32.55 |
|      chandelier     | 62.47 | 79.34 |
|        awning       | 27.03 | 33.04 |
|     streetlight     | 20.82 | 24.85 |
|        booth        | 55.05 | 57.18 |
| television receiver | 65.78 | 76.61 |
|       airplane      | 52.46 | 63.19 |
|      dirt track     |  4.39 | 18.76 |
|       apparel       | 28.42 | 38.05 |
|         pole        | 24.42 | 34.54 |
|         land        |  3.49 |  4.27 |
|      bannister      | 10.85 | 15.29 |
|      escalator      | 33.18 | 44.29 |
|       ottoman       |  38.7 | 54.25 |
|        bottle       | 32.29 | 47.27 |
|        buffet       | 44.63 | 47.34 |
|        poster       | 28.95 | 32.84 |
|        stage        | 15.02 | 20.96 |
|         van         | 41.73 | 56.04 |
|         ship        | 61.42 | 81.16 |
|       fountain      |  21.8 | 23.48 |
|    conveyer belt    | 81.22 | 91.07 |
|        canopy       | 13.44 | 14.61 |
|        washer       | 67.39 | 68.44 |
|      plaything      | 21.47 | 30.89 |
|    swimming pool    |  46.8 | 69.51 |
|        stool        | 34.71 | 48.97 |
|        barrel       | 26.06 | 41.47 |
|        basket       | 32.66 | 41.35 |
|      waterfall      | 41.38 |  58.2 |
|         tent        | 92.17 | 97.93 |
|         bag         |  10.8 | 13.61 |
|       minibike      | 53.83 | 66.26 |
|        cradle       | 79.72 | 95.29 |
|         oven        | 23.83 | 31.64 |
|         ball        | 43.86 | 61.95 |
|         food        | 51.63 | 61.47 |
|         step        |  7.9  |  9.79 |
|         tank        | 42.24 | 52.33 |
|      trade name     | 20.93 | 24.12 |
|      microwave      | 70.66 | 78.74 |
|         pot         | 35.76 | 41.42 |
|        animal       | 51.82 | 54.33 |
|       bicycle       | 52.46 | 73.88 |
|         lake        | 59.31 | 66.94 |
|      dishwasher     | 53.65 | 64.46 |
|        screen       | 74.33 | 86.88 |
|       blanket       | 10.22 | 11.86 |
|      sculpture      | 52.71 | 67.59 |
|         hood        | 63.81 | 70.12 |
|        sconce       |  40.4 | 52.79 |
|         vase        | 28.41 | 43.98 |
|    traffic light    | 36.28 |  48.2 |
|         tray        |  7.99 | 13.47 |
|        ashcan       | 36.02 | 45.89 |
|         fan         | 54.12 | 67.82 |
|         pier        | 48.55 | 85.02 |
|      crt screen     | 11.39 | 22.54 |
|        plate        | 48.29 | 58.27 |
|       monitor       | 39.29 | 46.57 |
|    bulletin board   | 37.18 | 49.71 |
|        shower       |  3.22 |  7.36 |
|       radiator      | 56.14 | 63.75 |
|        glass        |  6.48 |  6.76 |
|        clock        | 30.85 | 35.37 |
|         flag        |  34.0 | 38.66 |
+---------------------+-------+-------+

"""
table_B3 = """
+---------------------+-------+-------+
|        Class        |  IoU  |  Acc  |
+---------------------+-------+-------+
|         wall        | 74.88 |  87.6 |
|       building      | 80.24 |  91.0 |
|         sky         | 93.72 | 96.53 |
|        floor        | 79.89 | 91.13 |
|         tree        |  73.0 | 87.87 |
|       ceiling       |  83.3 | 91.47 |
|         road        | 81.56 | 88.57 |
|         bed         | 85.11 | 94.18 |
|      windowpane     | 58.89 | 76.64 |
|        grass        | 65.74 | 82.63 |
|       cabinet       | 56.28 | 70.71 |
|       sidewalk      | 63.83 | 79.95 |
|        person       | 77.78 | 90.25 |
|        earth        | 35.25 | 50.69 |
|         door        | 41.68 | 52.87 |
|        table        | 55.03 | 71.41 |
|       mountain      |  52.6 | 65.43 |
|        plant        | 48.37 | 61.56 |
|       curtain       |  70.2 | 82.42 |
|        chair        | 51.03 | 64.79 |
|         car         | 80.51 | 89.24 |
|        water        | 47.88 | 62.28 |
|       painting      | 70.42 | 86.73 |
|         sofa        | 58.11 | 75.21 |
|        shelf        | 40.33 | 60.68 |
|        house        |  39.6 | 50.08 |
|         sea         | 49.66 | 78.29 |
|        mirror       | 63.54 | 75.79 |
|         rug         | 62.08 | 69.23 |
|        field        | 29.63 | 43.86 |
|       armchair      | 38.09 | 56.53 |
|         seat        | 55.11 | 74.87 |
|        fence        | 40.75 | 57.41 |
|         desk        | 44.53 | 64.56 |
|         rock        | 44.35 | 69.22 |
|       wardrobe      |  43.7 | 63.25 |
|         lamp        | 58.16 | 70.89 |
|       bathtub       |  73.2 | 78.46 |
|       railing       |  31.2 | 44.45 |
|       cushion       | 53.11 | 67.24 |
|         base        | 20.77 |  32.4 |
|         box         | 20.95 | 26.35 |
|        column       | 44.31 | 59.13 |
|      signboard      | 33.11 | 47.51 |
|   chest of drawers  | 37.17 | 51.81 |
|       counter       | 30.99 | 35.32 |
|         sand        | 56.66 | 66.36 |
|         sink        | 67.26 | 77.46 |
|      skyscraper     | 63.06 | 78.42 |
|      fireplace      | 70.35 | 80.83 |
|     refrigerator    | 70.59 | 78.84 |
|      grandstand     | 37.39 | 68.15 |
|         path        | 23.05 | 32.23 |
|        stairs       | 28.13 | 35.41 |
|        runway       |  70.1 | 85.72 |
|         case        | 46.32 | 58.69 |
|      pool table     | 91.06 | 95.21 |
|        pillow       | 55.47 | 68.31 |
|     screen door     | 64.03 | 70.38 |
|       stairway      | 28.43 | 38.32 |
|        river        |  9.44 | 17.54 |
|        bridge       | 44.42 |  53.8 |
|       bookcase      | 36.23 | 49.33 |
|        blind        | 40.44 | 45.66 |
|     coffee table    | 56.97 | 76.23 |
|        toilet       |  82.3 | 88.37 |
|        flower       | 33.66 | 54.59 |
|         book        | 45.73 | 66.65 |
|         hill        | 12.35 | 17.37 |
|        bench        | 40.54 | 46.42 |
|      countertop     | 49.06 | 63.37 |
|        stove        | 72.52 | 78.22 |
|         palm        | 50.74 | 69.85 |
|    kitchen island   | 38.64 | 62.93 |
|       computer      | 58.94 | 68.08 |
|     swivel chair    | 41.91 | 58.13 |
|         boat        | 50.08 | 60.15 |
|         bar         |  29.3 |  34.8 |
|    arcade machine   | 58.24 | 60.64 |
|        hovel        | 13.37 | 15.19 |
|         bus         | 87.55 | 93.74 |
|        towel        | 55.07 | 64.75 |
|        light        | 49.65 | 59.85 |
|        truck        |  17.5 | 31.39 |
|        tower        | 28.18 | 48.64 |
|      chandelier     |  61.9 | 78.27 |
|        awning       |  28.0 | 34.26 |
|     streetlight     | 21.59 | 26.48 |
|        booth        | 50.59 | 52.39 |
| television receiver | 63.48 | 73.75 |
|       airplane      | 52.03 | 63.84 |
|      dirt track     |  4.28 | 18.83 |
|       apparel       | 29.22 | 40.17 |
|         pole        | 23.26 | 32.67 |
|         land        |  4.93 |  6.46 |
|      bannister      |  9.43 | 13.97 |
|      escalator      | 35.66 | 48.41 |
|       ottoman       | 40.39 | 53.72 |
|        bottle       | 30.56 | 40.96 |
|        buffet       | 47.43 | 51.39 |
|        poster       | 28.46 | 32.97 |
|        stage        | 13.88 | 20.63 |
|         van         | 42.64 | 56.81 |
|         ship        | 64.51 | 82.34 |
|       fountain      | 20.91 | 22.82 |
|    conveyer belt    | 80.29 | 90.73 |
|        canopy       | 17.02 | 19.14 |
|        washer       |  70.4 | 73.78 |
|      plaything      | 20.75 | 30.23 |
|    swimming pool    | 54.45 | 80.46 |
|        stool        | 33.86 | 45.41 |
|        barrel       | 25.99 | 29.49 |
|        basket       | 32.66 | 40.39 |
|      waterfall      | 37.11 | 52.51 |
|         tent        | 84.11 |  98.3 |
|         bag         | 10.72 | 13.86 |
|       minibike      | 52.54 | 65.35 |
|        cradle       | 78.22 | 95.32 |
|         oven        | 26.18 | 37.26 |
|         ball        | 45.83 | 60.49 |
|         food        | 47.82 | 55.92 |
|         step        |  6.82 |  8.64 |
|         tank        | 40.51 | 51.27 |
|      trade name     | 19.03 | 21.69 |
|      microwave      | 66.39 | 73.76 |
|         pot         | 35.82 | 42.85 |
|        animal       | 51.63 | 53.68 |
|       bicycle       | 53.07 | 74.47 |
|         lake        | 54.09 | 69.78 |
|      dishwasher     | 54.35 | 70.74 |
|        screen       | 72.71 | 84.68 |
|       blanket       |  7.76 |  8.94 |
|      sculpture      | 53.43 | 76.63 |
|         hood        | 63.46 | 69.05 |
|        sconce       | 38.96 | 50.35 |
|         vase        | 26.14 | 43.24 |
|    traffic light    |  36.0 | 48.59 |
|         tray        |  9.01 | 13.73 |
|        ashcan       |  37.9 | 49.04 |
|         fan         | 54.03 | 69.84 |
|         pier        | 53.33 |  86.1 |
|      crt screen     |  12.9 | 26.99 |
|        plate        | 49.88 | 61.57 |
|       monitor       | 33.56 | 40.09 |
|    bulletin board   | 40.62 | 58.13 |
|        shower       |  4.83 |  6.47 |
|       radiator      | 53.54 |  58.6 |
|        glass        |  8.05 |  8.72 |
|        clock        | 31.33 |  35.5 |
|         flag        | 40.13 |  46.0 |
+---------------------+-------+-------+

"""

# Comparing average IoU values from tables A and B and returning the top 3 differences
top_avg_differences = compare_average_iou_values([table_A1, table_A2, table_A3], [table_B1, table_B2, table_B3], k=10)
print(top_avg_differences)
