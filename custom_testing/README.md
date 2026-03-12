# custom_testing/ — Build Your Own Test Data

This folder is where you put your own CSV test files to run through the trained model.

---

## Step-by-Step Process

### Step 1: Find your model's log folder
After training, a folder is created automatically:
```
logs/vrp10-YYYY-MM-DD_HH-MM-SS/
```

### Step 2: Open `model_info.txt` in that folder
```
logs/vrp10-.../model_info.txt
```
This file contains:
- Exact CSV column layout for this model
- Constraints (capacity, demand range, coordinate limits)
- Copy-paste terminal commands
- A ready-to-use example CSV row

### Step 3: Build your CSV
Copy `example_vrp10.csv` (or `example_tsp10.csv`) from this folder and modify it:
- Each **row** = one delivery scenario (one "problem")
- Each row must have exactly the right number of columns (see model_info.txt)
- Coordinates must be normalized to **[0.0, 1.0]**
- For VRP: last node per row must be the **depot** with demand=0

### Step 4: Run inference with your CSV
```powershell
# Replace <LOG> with your actual log folder name
python view_routes.py --is_train=False --task=vrp10 `
    --load_path "logs/<LOG>/model/model.ckpt-10000" `
    --csv_path custom_testing/my_deliveries.csv

# Example with the provided template:
python view_routes.py --is_train=False --task=vrp10 `
    --csv_path custom_testing/example_vrp10.csv
```

### Step 5: Find your results
Everything is saved inside the model's log folder:
```
logs/vrp10-.../
├── routes.txt   ← Driver instructions (all problems)
├── routes.png   ← Visual route map
└── model_info.txt
```

---

## How to Normalize Coordinates

If you have real GPS coordinates, normalize them to [0.0, 1.0]:

```python
import numpy as np

lats = [28.61, 28.70, 28.55, ...]  # your latitudes
lons = [77.20, 77.10, 77.35, ...]  # your longitudes

lat_min, lat_max = min(lats), max(lats)
lon_min, lon_max = min(lons), max(lons)

x_norm = [(lon - lon_min) / (lon_max - lon_min) for lon in lons]
y_norm = [(lat - lat_min) / (lat_max - lat_min) for lat in lats]
```

---

## VRP10 CSV Format Quick Reference

| Column group | Columns | Values |
|---|---|---|
| problem_id | 1 column | Any integer (1, 2, 3...) |
| C1 | C1_x, C1_y, C1_demand | x∈[0,1], y∈[0,1], demand∈[1..9] |
| C2–C10 | Same pattern × 9 | Same constraints |
| DEPOT | depot_x, depot_y, depot_demand | x∈[0,1], y∈[0,1], **demand=0** |

**Total: 34 columns** (1 id + 10 customers × 3 + 1 depot × 3)

---

## TSP10 CSV Format Quick Reference

| Column group | Columns | Values |
|---|---|---|
| problem_id | 1 column | Any integer |
| C1–C10 | C1_x, C1_y × 10 | x∈[0,1], y∈[0,1] |

**Total: 21 columns** (1 id + 10 cities × 2)  
No depot needed for TSP.
