import geopandas as gpd
import numpy as np


def crop_groups(gdf):
    cat="reported_crop_name"
    CROP_GROUPS = {
        "Lettuce": ["Lettuce, Baby", "Lettuce, Baby (Fall/Winter)", "Lettuce, Baby (Spring/Summer)", "Lettuce, Head", "Lettuce, Head (Fall/Winter)", "Lettuce, Head (Spring/Summer)", "Lettuce, Iceberg", "Lettuce, Iceberg (Spring/Summer)", "Lettuce, Leaf", "Lettuce, Leaf (Fall/Winter)", "Lettuce, Leaf (Spring/Summer)", "Lettuce, Romaine", "Lettuce, Romaine (Fall/Winter)", "Lettuce, Romaine (Spring/Summer)", "Lettuce, Romaine Hearts", "Lettuce, Romaine Hearts (Fall/Winter)", "Other - Green Butter", "Other - Green Butter Lettuce", "Other - Lettuce, Butter", "Spring Mix", "Spring Mix (multiple cuttings)", "Spring Mix, Baby", "Spring Mix, Baby (multiple cuttings)"],
        "Spinach": ["Spinach, Baby", "Spinach, Baby (multiple cuttings)", "Spinach, Bunch", "Spinach, Clip", "Spinach, Clip (multiple cuttings)"],
        "Broccoli": ["Broccoli", "Broccoli (Fall/Winter)", "Broccoli (Spring/Summer)", "Broccoli Rabe", "Broccoli, Seed", "Broccolini"],
        "Cauliflower": ["Cauliflower", "Cauliflower (Fall/Winter)", "Cauliflower (Spring/Summer)"],
        "Celery": ["Celery", "Other - Chinese Celery"],
        "Strawberry": ["Strawberry", "Strawberry, 2-step program, 1st step", "Strawberry, Greater than 12-months variety", "Strawberry, Up to 12-months variety"],
        "Wine Grapes": ["Grapes - Wine"],
        "Other Brassicas": ["Broccolette", "Brussels Sprouts", "Cabbage", "Cabbage Green", "Cabbage Red", "Cabbage, Chinese", "Cabbage, Napa", "Cabbage, Savoy", "Kohlrabi"],
        "Leafy Greens": ["Amaranth", "Arugula", "Arugula, Baby", "Arugula, Wild", "Bok Choy", "Bok Choy, Baby", "Chard, Green", "Chard, Red", "Chard, Swiss", "Chinese Greens (A Choy)", "Chinese Greens (Bok Choy)", "Chinese Greens (Bok Choy, Baby)", "Chinese Greens (Gai Choy)", "Chinese Greens (Gai Lan)", "Chinese Greens (On Choy)", "Chinese Greens (Shanghai Bok Choy)", "Chinese Greens (Snow Pea Tips)", "Chinese Greens (Tong Ho)", "Chinese Greens (Yam Leaves)", "Chinese Greens (Yu Choy)", "Collard Greens", "Cress", "Dandelion Greens", "Other - Dandelion", "Endive", "Escarole", "Frisee", "Radicchio", "Kale", "Kale (multiple cuttings)", "Kale, Baby", "Kale, Baby (multiple cuttings)", "Mixed Greens", "Mixed Greens, Baby", "Mizuna", "Mustard", "Mustard, Baby", "Other - Fenugreek leaves", "Other - Methi Leaf (Fenugreek)", "Other - Ong Choy", "Other - Tong Ho", "Watercress"],
        "Herbs": ["Anise", "Cilantro", "Cilantro, Bunch", "Cilantro, Clip", "Dill", "Fennel", "Herb Savory", "Mint", "Parsley"],
        "Roots & Bulbs": ["Beet", "Carrots", "Carrots, Baby", "Carrots, Full-Sized", "Celeriac", "Garlic", "Leek", "Onions, Green", "Onions, dry", "Radish", "Turnip"],
        "Fruiting & Other Vegetables": ["Asparagus", "Corn - Sweet", "Cucumbers", "Peppers, Bell", "Peppers, Bell, Multiple Harvest Variety", "Peppers, Bell, Single Harvest Variety", "Peppers, Chili", "Pimiento", "Pumpkin", "Squash, Summer", "Squash, Winter", "Tomato", "Tomatoes, Fresh Market"],
        "Legumes": ["Beans", "Beans, Lima", "Beans, Seed", "Beans, dry", "Beans, dry - Garbanzo", "Beans, dry - Lima", "Beans, green (snap beans)", "Peas, Seed", "Peas, Snap or Sugar"],
        "Other Berries": ["Blackberry", "Blueberry", "Raspberry"],
        "Tree Crops": ["Apricots", "Avocados", "Cherry", "Olives", "Walnuts"],
        "Field & Forage": ["Alfalfa", "Barley - Grain", "Corn", "Ryegrass, Winter", "Seed Crops", "Turf (sod/grass)"],
        "Cover Crops": ["Cover Crop, Legume (Irrigated)", "Cover Crop, Non-Legume (Irrigated)", "Cover Crop, Non-Legume (Non-Irrigated)", "Cover Crop, RSCAVENGE (Irrigated)"],
        "Nursery & Ornamental": ["Flowers", "Other - Flowers", "Roses", "Other - Cosmo", "Nursery Trees", "Other - Christmas Trees", "Other - Monterey pine Christmas tree", "Other - Eucalyptus", "Other - Nursery plants  mainly trees", "Other - Ornamental Crops grown in containers", "Other - Ornamentals in containers", "Other - Potted Nursery Plants", "Other - Potted Plants"],
    }
    crop_category_dict = {crop: group for group, crops in CROP_GROUPS.items() for crop in crops}
    return gdf[cat].map(crop_category_dict)