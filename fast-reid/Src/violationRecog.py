from torch import Tensor, tensor, cat, isin, unique
from cv2 import line
from numpy import ndarray
import numpy as np

def height_TL (line_height:list, new_height:float) -> float:
   line_height['n'] += 1
   line_height['x'] = new_height
   line_height['sum'] += new_height
   line_height['mean'] = line_height['sum']/line_height['n']
   return line_height['mean']

def place_line (image:ndarray, position:float):
   position = round(position)
   line(image, (0,position), (image.shape[1],position), (0,255,0), 3)

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate perpendicular distance from point to line segment"""
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:  # Line is a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    param = dot / len_sq
    
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    
    return np.sqrt((px - xx)**2 + (py - yy)**2)

def recognize_violation_free_line(image:ndarray, vehicles:Tensor, chosen_xyxy:Tensor, line_points:list, veh_set:Tensor, threshold:float=15.0):
    """
    Free-form line violation detection
    
    Args:
        image: Current frame
        vehicles: Detected vehicles tensor
        chosen_xyxy: Selected traffic light coordinates (kept for compatibility)
        line_points: List of (x1, y1, x2, y2) line segments
        veh_set: Set of vehicle IDs that have violated
        threshold: Distance threshold for violation detection
    """
    
    if not line_points:
        return (0, 0, veh_set, vehicles[0:0])  # Return empty if no lines drawn
    
    # Draw all line segments
    for x1, y1, x2, y2 in line_points:
        line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    violators = []
    
    # Check each vehicle against all line segments
    for i, vehicle in enumerate(vehicles):
        # Get vehicle center point
        center_x = (vehicle[0] + vehicle[2]) / 2
        center_y = (vehicle[1] + vehicle[3]) / 2
        
        # Check distance to each line segment
        violated = False
        for x1, y1, x2, y2 in line_points:
            distance = point_to_line_distance(center_x.item(), center_y.item(), x1, y1, x2, y2)
            if distance <= threshold:
                violators.append(vehicle)
                violated = True
                break  # Found violation, no need to check other lines
    
    if violators:
        violator_tensor = cat(violators) if len(violators) > 1 else violators[0].unsqueeze(0)
        # Update vehicle set with new violators
        veh_set = unique(cat((veh_set, violator_tensor[:, 4])))
        
        # Get all vehicles that are in the violation set
        all_violators = vehicles[isin(vehicles[:,4], veh_set)]
        return (0, 0, veh_set, all_violators)
    
    return (0, 0, veh_set, vehicles[0:0])  # Return empty tensor if no violations

# Keep original function for backward compatibility
def recognize_violation (image:ndarray, vehicles:Tensor, chosen_xyxy:Tensor, manual_line_y:float, veh_set:Tensor, imaginary_line:bool=False):
   """
   Original function maintained for compatibility
   """
   position = manual_line_y
   
   if imaginary_line: 
      place_line(image, position)
   
   new_height = chosen_xyxy[3].item() - chosen_xyxy[1].item()
   raw_position = 3.5 * new_height + chosen_xyxy[1].item()
   
   violators = vehicles[(vehicles[:,1]+(vehicles[:,3]-vehicles[:,1])/2) > position]
   veh_set = unique(cat((veh_set, violators[:, 4])))
   violators = vehicles[isin(vehicles[:,4], veh_set)]
   current_violators = violators[(violators[:,1]+(violators[:,3]-violators[:,1])/2) < position]
   
   return (raw_position, position, veh_set, current_violators)