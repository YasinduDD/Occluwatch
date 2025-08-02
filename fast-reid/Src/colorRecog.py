from numpy import array, ndarray
from cv2 import cvtColor, inRange, countNonZero
from torch import ceil, floor, Tensor
from typing import Dict, List, Tuple

def maskColor (frame_area:ndarray, validateYellow:bool=False):
   maskRed     = inRange(frame_area, array([  0,100,100]), array([ 10,255,255]))
   maskRed    += inRange(frame_area, array([160,100,100]), array([179,255,255]))

   if validateYellow:
      maskYellow = maskRed + inRange(frame_area, array([ 10,100,100]), array([ 40,255,255]))
      yellow = countNonZero(maskYellow)
      return yellow

   else:
      maskYellow = inRange(frame_area, array([ 10,100,100]), array([ 40,255,255]))
      maskGreen  = inRange(frame_area, array([ 40,100,100]), array([ 80,255,255]))
      red      = countNonZero(maskRed)
      yellow   = countNonZero(maskYellow)
      green    = countNonZero(maskGreen)

      return (red, yellow, green)
   pass

def validateYellow (cropped_frame:ndarray) -> bool:
   dim = list(cropped_frame.shape[:2])
   cen = (round(dim[0]/2),round(dim[1]/2))

   crop = lambda frm, cen, dim: frm[cen[0]-dim[0]//2:cen[0]+dim[0]//2, cen[1]-dim[1]//2:cen[1]+dim[1]//2]

   if dim[0]>dim[1]: dim[0] = dim[0]//4   # Vertical Traffic Light
   else: dim[1] = dim[1]//4               # Horizontal Traffic Light
   
   yellow_light = crop(cropped_frame, cen, dim)

   if maskColor(yellow_light, validateYellow=True)>50: return True
   else: return False


def recognize_color (frame:ndarray, prediction:Tensor, print_info:bool=False):
   color = {
      'red'          : [],
      'yellow'       : [],
      'green'        : [],
      "light's off"  : []
   }
   frame = cvtColor(frame, 40)

   for n, traffic_light in enumerate(prediction):
      x1 = int(ceil(traffic_light[0]))
      y1 = int(ceil(traffic_light[1]))
      x2 = int(floor(traffic_light[2]))
      y2 = int(floor(traffic_light[3]))
      confidence  = round(float(traffic_light[5]*100), 2)
      traffic_light = frame[y1:y2,x1:x2]

      red, yellow, green = maskColor(traffic_light)

      if (red>green) & (red>yellow) & (red>50):
         if not validateYellow(traffic_light): color["red"].append((n, confidence))
         else: color["yellow"].append((n, confidence))
      elif (green>yellow) & (green>50):
         color["green"].append((n, confidence))
      elif yellow>50:
         if validateYellow(traffic_light): color["yellow"].append((n, confidence))
         else: color["red"].append((n, confidence))
      else:
         color["light's off"].append((n, confidence))

      if print_info: print(traffic_light.shape[:2], f'\nRED: {red}', f'\tYELLOW: {yellow}', f'\tGREEN: {green}')

   if print_info: print(color)
   return color

def chooseOne(light_colors: Dict[str, List[Tuple[int, float]]]) -> Tuple[int, float]:
    chosen = [0, 0.0]
    for color, values in light_colors.items():
        if color != "red": continue
        for value in values:
            if value[-1] > chosen[-1]:
                chosen = list(value)
    return tuple(chosen)