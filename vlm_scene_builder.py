# By dreamraster · dreaMSCend
import os
import json
import random
import uuid
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_SIZE = 1000
OUTPUT_DIR = "datasets/scene_interaction_v1"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
JSONL_PATH = os.path.join(OUTPUT_DIR, "dataset.jsonl")

# Scene dimensions (normalized to 1000x1000 for the model)
CANVAS_SIZE = 1000
SHAPE_SIZE = 60

COLORS = ["red", "blue", "green", "yellow", "purple", "cyan"]
SHAPES = ["rectangle", "circle"]

os.makedirs(IMAGE_DIR, exist_ok=True)

def generate_sample(index):
    """Generate a single 2D scene sample:
    1. Randomly place 2-4 existing objects.
    2. pick a new object type/color/target position.
    3. pick an optional connection target.
    4. Render image.
    5. Construct reasoning and JSON action.
    """
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw simple grid for visual reference
    for i in range(0, CANVAS_SIZE, 100):
        draw.line([(i, 0), (i, CANVAS_SIZE)], fill=(220, 220, 220), width=1)
        draw.line([(0, i), (CANVAS_SIZE, i)], fill=(220, 220, 220), width=1)

    existing_objects = []
    num_existing = random.randint(1, 3)
    
    for i in range(num_existing):
        obj_id = f"obj_{i:03d}"
        color = random.choice(COLORS)
        shape = random.choice(SHAPES)
        x = random.randint(100, 900)
        y = random.randint(100, 900)
        
        # Draw the object
        bbox = [x - SHAPE_SIZE//2, y - SHAPE_SIZE//2, x + SHAPE_SIZE//2, y + SHAPE_SIZE//2]
        if shape == "rectangle":
            draw.rectangle(bbox, fill=color, outline="black")
        else:
            draw.ellipse(bbox, fill=color, outline="black")
            
        existing_objects.append({
            "id": obj_id,
            "type": shape,
            "color": color,
            "position": [x, y]
        })

    # New action: Place and Connect
    target_color = random.choice([c for c in COLORS if c not in [o["color"] for o in existing_objects]])
    target_shape = random.choice(SHAPES)
    target_pos = [random.randint(100, 900), random.randint(100, 900)]
    
    # Connection target
    conn_obj = random.choice(existing_objects)
    
    instruction = (
        f"Place a {target_color} {target_shape} at {target_pos} and "
        f"connect it to the {conn_obj['color']} {conn_obj['type']} at {conn_obj['position']}."
    )
    
    # Construct reasoning
    reasoning = (
        f"The scene contains {num_existing} objects. "
        f"I need to place a new {target_color} {target_shape} at coordinates {target_pos}. "
        f"Then, I will establish a mechanical connection between the new object and {conn_obj['id']} "
        f"(the {conn_obj['color']} {conn_obj['type']})."
    )
    
    # The outcome state (for the JSON solution)
    solution = {
        "action": "PLACE_AND_CONNECT",
        "object": target_shape,
        "color": target_color,
        "position": target_pos,
        "connect_to": conn_obj["id"]
    }
    
    # Save image
    img_filename = f"scene_{index:05d}_{uuid.uuid4().hex[:8]}.png"
    img_path = os.path.join(IMAGE_DIR, img_filename)
    img.save(img_path)
    
    return {
        "image": os.path.abspath(img_path),
        "instruction": instruction,
        "reasoning": reasoning,
        "solution": json.dumps(solution),
        "state_metadata": {
            "existing": existing_objects,
            "new": solution
        }
    }

print(f"Generating {DATASET_SIZE} samples in {OUTPUT_DIR}...")
with open(JSONL_PATH, "w") as f:
    for i in range(DATASET_SIZE):
        sample = generate_sample(i)
        f.write(json.dumps(sample) + "\n")
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{DATASET_SIZE}")

print(f"Done! Dataset saved to {JSONL_PATH}")
