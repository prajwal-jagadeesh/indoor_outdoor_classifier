import os
import shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# === Paths ===
image_dir = 'shard_94/images'
indoor_dir = os.path.join(image_dir, 'indoor')
outdoor_dir = os.path.join(image_dir, 'outdoor')
doubt_dir = os.path.join(image_dir, 'doubt')
os.makedirs(indoor_dir, exist_ok=True)
os.makedirs(outdoor_dir, exist_ok=True)
os.makedirs(doubt_dir, exist_ok=True)

# === Log File ===
log_file_path = 'classification_log.txt'
with open(log_file_path, 'w') as log_file:
    log_file.write('Filename, Label, Confidence, Scene Category\n')  # Header

# === Load Labels ===
with open('places365/categories_places365.txt') as f:
    classes = [line.strip().split(' ')[0][3:] for line in f]

# === Map categories to indoor/outdoor ===
# Expanding the mapping to include more keywords
# Categorized indoor and outdoor locations
indoor_keywords = [
    'airplane_cabin', 'airport_terminal', 'alcove', 'amphitheater', 'amusement_arcade', 'archive',
    'art_gallery', 'art_school', 'art_studio', 'artists_loft', 'assembly_line',
    'attic', 'auditorium', 'auto_factory', 'auto_showroom', 'ball_pit',
    'ballroom', 'banquet_hall', 'bank_vault', 'basement', 'basketball_court/indoor',
    'bazaar/indoor', 'bathroom', 'beauty_salon', 'bedroom', 'beer_hall',
    'berth', 'biology_laboratory', 'bookstore', 'booth/indoor', 'bow_window/indoor',
    'bowling_alley', 'boxing_ring', 'bus_interior', 'bus_station/indoor',
    'butchers_shop', 'cafeteria', 'candy_store', 'car_interior', 'chemistry_lab', 'church/indoor', 
    'childs_room', 'classroom', 'clean_room', 'closet', 'clothing_store', 'cockpit',
    'coffee_shop', 'computer_room', 'conference_room', 'corridor', 'department_store',
    'dining_hall', 'dining_room', 'dorm_room', 'dressing_room', 'drugstore', 'elevator/door', 
    'elevator_lobby', 'elevator_shaft', 'engine_room', 'entrance_hall',
    'escalator/indoor', 'fabric_store', 'fastfood_restaurant', 'fire_escape', 'flea_market/indoor',
    'florist_shop/indoor', 'galley', 'garage/indoor', 'general_store/indoor', 'gift_shop',
    'greenhouse/indoor', 'gymnasium/indoor', 'hangar/indoor', 'hardware_store', 'home_office',
    'home_theater', 'hospital_room', 'hotel_room', 'ice_skating_rink/indoor', 'interior',
    'jacuzzi/indoor', 'jail_cell', 'jewelry_shop', 'kindergarden_classroom', 'kitchen',
    'laundromat', 'lecture_room', 'library/indoor', 'living_room', 'lobby',
    'locker_room', 'market/indoor', 'martial_arts_gym', 'medina', 'mezzanine', 'movie_theater/indoor',
    'museum/indoor', 'music_studio', 'office', 'office_building', 'office_cubicles',
    'operating_room', 'pantry', 'parking_garage/indoor', 'pet_shop', 'performance', 'pharmacy', 
    'phone_booth', 'physics_laboratory', 'playroom', 'pub/indoor', 'reception', 'recreation_room',
    'repair_shop', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rodeo',
    'schoolhouse', 'science_museum', 'server_room', 'shoe_shop', 'shop',
    'shopfront', 'shopping_mall/indoor', 'stage/indoor', 'staircase', 'storage_room', 'subway_station/platform',
    'supermarket', 'sushi_bar', 'swimming_pool/indoor', 'television_room', 'television_studio', 'throne_room',
    'toyshop', 'train_interior', 'utility_room', 'veterinarians_office', 'waiting_room'
]


outdoor_keywords = [
    'airfield', 'alley', 'amusement_park', 'apartment_building/outdoor',
    'aquarium', 'aqueduct', 'arcade', 'arch', 'archaelogical_excavation',
    'army_base', 'asia', 'athletic_field/outdoor', 'badlands', 'bamboo_forest',
    'banana_plantation', 'bar', 'barn', 'barndoor', 'baseball',
    'baseball_field', 'basketball', 'bazaar/outdoor', 'beach', 'beach_house',
    'bedchamber', 'beer_garden', 'boardwalk', 'boathouse', 'boat_deck', 'botanical_garden',
    'broadleaf', 'bridge', 'building_facade', 'bullring', 'burial_chamber',
    'butte', 'cabin/outdoor', 'campus', 'campsite',
    'canyon', 'carrousel', 'castle', 'catacomb', 'cemetery',
    'chalet', 'church/outdoor', 'cliff', 'coast',
    'conference_center', 'construction_site', 'corn_field', 'corral', 'cottage',
    'courthouse', 'courtyard', 'creek', 'crevasse', 'crosswalk',
    'cultivated', 'dam', 'delicatessen', 'desert_road', 'diner/outdoor',
    'discotheque', 'door', 'doorway/outdoor', 'downtown', 'driveway',
    'embassy', 'excavation', 'exterior', 'farm', 'field_road',
    'fire_station', 'fishpond', 'food_court', 'football', 'football_field',
    'forest_path', 'forest_road', 'formal_garden', 'fountain', 'garage/outdoor',
    'gas_station', 'general_store/outdoor', 'glacier', 'golf_course', 'greenhouse/outdoor',
    'grotto', 'hangar/outdoor', 'harbor', 'hayfield', 'heliport',
    'highway', 'hotel/outdoor', 'house', 'hunting_lodge/outdoor', 'ice_cream_parlor',
    'ice_floe', 'ice_shelf', 'ice_skating_rink/outdoor', 'iceberg', 'igloo',
    'industrial_area', 'inn/outdoor', 'islet', 'japanese_garden', 'junkyard',
    'kasbah', 'kennel/outdoor', 'lagoon', 'landing_deck', 'landfill',
    'lawn', 'legislative_chamber', 'library/outdoor', 'lighthouse', 'loading_dock',
    'lock_chamber', 'manufactured_home', 'mansion', 'market/outdoor', 'marsh',
    'mausoleum', 'mosque/outdoor', 'mountain', 'mountain_path', 'mountain_snowy',
    'movie_theater/outdoor', 'motel', 'museum/outdoor', 'natural', 'natural_history_museum',
    'nursery', 'nursing_home', 'oast_house', 'ocean', 'ocean_deep',
    'oilrig', 'orchard', 'orchestra_pit', 'pagoda', 'palace',
    'park', 'parking_garage/outdoor', 'parking_lot', 'pasture', 'patio',
    'pavilion', 'picnic_area', 'pier',
    'pizzeria', 'platform', 'plaza', 'playground', 'pond',
    'porch', 'promenade', 'public', 'racecourse', 'raceway',
    'raft', 'railroad_track', 'rainforest', 'residential_neighborhood', 'rice_paddy',
    'river', 'rock_arch', 'roof_garden', 'rope_bridge', 'ruin',
    'runway', 'sandbox', 'sauna', 'shed', 'shower',
    'shopping_mall/outdoor', 'ski_resort', 'ski_slope', 'sky', 'skyscraper',
    'slum', 'snowfield', 'soccer', 'soccer_field', 'stage/outdoor',
    'stable', 'street', 'subway', 'swamp',
    'swimming_hole', 'swimming_pool/outdoor', 'synagogue/outdoor', 'ticket_booth', 'topiary_garden',
    'tower', 'trench', 'tree_farm', 'tree_house', 'tundra',
    'urban', 'valley', 'vegetable_garden', 'vegetation', 'viaduct',
    'village', 'vineyard', 'volcano', 'volleyball_court/outdoor', 'water',
    'water_park', 'water_tower', 'waterfall', 'watering_hole', 'wave',
    'wet_bar', 'wheat_field', 'wild', 'wind_farm', 'windmill',
    'yard', 'youth_hostel', 'zen_garden'
]

def classify_scene(category):
    '''
    Classifies scenes into indoor, outdoor, or doubt based on the category.
    '''
    # Check for indoor
    for word in indoor_keywords:
        if word in category:
            return 'indoor'
    
    # Check for outdoor
    for word in outdoor_keywords:
        if word in category:
            return 'outdoor'
    
    return 'doubt'

# === Load Model ===
def load_places365_model():
    '''
    Loads the Places365 model (ResNet18).
    '''
    model_file = 'places365/resnet18_places365.pth.tar'
    model = models.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_places365_model()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Confidence Threshold ===
confidence_threshold = 0.05  # Set confidence threshold to 0.10

# === Classify Images and Log Results ===
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_dir, filename)
        try:
            # Open image
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            # Get model prediction
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                top_prob, pred_idx = torch.max(probs, 1)
                category = classes[pred_idx.item()]
                confidence = top_prob.item()

                # Classify based on the scene category and confidence threshold
                if confidence < confidence_threshold:
                    label = 'doubt'
                else:
                    label = classify_scene(category)

            # Move image based on label
            if label == 'indoor':
                target_dir = indoor_dir
            elif label == 'outdoor':
                target_dir = outdoor_dir
            else:
                target_dir = doubt_dir

            shutil.move(img_path, os.path.join(target_dir, filename))
            print(f'{filename} → {label} ({confidence:.2f}) - Scene: {category}')

            # Log to text file
            with open(log_file_path, 'a') as log_file:
                log_file.write(f'{filename}, {label}, {confidence:.2f}, {category}\n')

        except Exception as e:
            print(f'Error processing {filename}: {e}')
            with open(log_file_path, 'a') as log_file:
                log_file.write(f'{filename}, error, -1.00, {str(e)}\n')  # Log error if any

