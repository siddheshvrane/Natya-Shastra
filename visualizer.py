import cv2
import numpy as np
import math

class Visualizer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.current_landmarks = None
        self.phi = 1.61803398875 # Golden Ratio

    def update(self, landmarks):
        """Update the state with new landmarks."""
        self.current_landmarks = landmarks

    def get_pt(self, idx):
        """Helper to get (x, y) from landmark index."""
        if not self.current_landmarks: return (0, 0)
        lm = self.current_landmarks[idx]
        return (int(lm.x * self.width), int(lm.y * self.height))

    def draw(self, image):
        """Draw mathematical effects on the image."""
        if not self.current_landmarks:
            return image
            
        # 1. Draw Bounding Boxes (The "Analytic" Layer)
        image = self.draw_all_bounding_boxes(image)

        # 2. Draw Pingala/Parabolic Flows (The "Organic" Layer)
        image = self.draw_pingala_flows(image)
        
        return image

    def draw_bounding_box(self, image, indices, color=(255, 255, 0), label_prefix=""):
        """Draw a bounding box around the set of landmark indices."""
        pts = [self.get_pt(i) for i in indices]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Add some padding
        pad = 10
        min_x -= pad
        max_x += pad
        min_y -= pad
        max_y += pad
        
        # 1. Main Rect (Thin Blue/Cyber line) - corners only or full rect? 
        # User wants "bounding box calculations". Let's do a full rect but stylized.
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (200, 100, 0), 1)
        
        
        # 3. Coordinates Text (Yellow) - As requested "x: ... y: ..."
        # Place them at corners
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_color = (100, 255, 255) # Yellowish
        
        # Top-Left coordinate
        cv2.putText(image, f"x: {min_x} y: {min_y}", (min_x, min_y - 5), font, font_scale, font_color, 1)
        
        # Sometimes show center or dimensions? Let's stick to corners as per ref image
        # cv2.putText(image, f"w: {max_x-min_x}", (min_x, max_y + 15), font, font_scale, font_color, 1)

        return image

    def draw_all_bounding_boxes(self, image):
        # Define groups
        # Face: 0-10
        # Torso: 11, 12, 23, 24
        # Left Arm: 11, 13, 15, 17, 19, 21
        # Right Arm: 12, 14, 16, 18, 20, 22
        # Left Leg: 23, 25, 27, 29, 31
        # Right Leg: 24, 26, 28, 30, 32
        
        self.draw_bounding_box(image, range(0, 11), label_prefix="Face") # Face
        self.draw_bounding_box(image, [11, 12, 23, 24], label_prefix="Torso") # Torso
        
        # Extremity boxes (Hands/Feet) are more interesting than whole arms
        self.draw_bounding_box(image, [15, 17, 19, 21], label_prefix="L-Hand") # L Hand
        self.draw_bounding_box(image, [16, 18, 20, 22], label_prefix="R-Hand") # R Hand
        self.draw_bounding_box(image, [27, 29, 31], label_prefix="L-Foot") # L Foot
        self.draw_bounding_box(image, [28, 30, 32], label_prefix="R-Foot") # R Foot
        
        return image

    def draw_parabola_curve(self, image, p1, p2, color=(255, 255, 255), thickness=1, dashed=True):
        """Draws a 'Pingala style' parabolic curve between p1 and p2."""
        dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        if dist == 0: return image

        # The Golden Ratio Control Point
        # Instead of a simple midpoint, we bias it and push it out based on phi
        
        # Midpoint
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        
        # Vector p1 -> p2
        vx, vy = p2[0] - p1[0], p2[1] - p1[1]
        
        # Perpendicular vector
        px, py = -vy, vx
        
        # Amplitude based on distance and Golden Ratio
        # We want the curve to look "sweeping"
        amplitude = dist * (1 / self.phi) * 0.5
        
        # Normalize perp vector
        norm = math.hypot(px, py)
        if norm == 0: norm = 1
        px /= norm
        py /= norm
        
        # Control point C
        cx = mx + px * amplitude
        cy = my + py * amplitude
        
        # Generate points for Quadratic Bezier
        pts = []
        steps = int(dist / 5) + 10 # Dynamic resolution
        for t in np.linspace(0, 1, steps):
            bx = (1-t)**2 * p1[0] + 2*(1-t)*t * cx + t**2 * p2[0]
            by = (1-t)**2 * p1[1] + 2*(1-t)*t * cy + t**2 * p2[1]
            pts.append((int(bx), int(by)))

        # Draw DASHED or Solid
        if dashed:
            for i in range(0, len(pts)-1, 2): # Draw every other segment
                cv2.line(image, pts[i], pts[i+1], color, thickness)
        else:
            for i in range(len(pts)-1):
                cv2.line(image, pts[i], pts[i+1], color, thickness)
                
        # Draw arrow at the end to show flow
        # cv2.arrowedLine(image, pts[-2], pts[-1], color, thickness, tipLength=0.5) 
        # (Arrow might look messy on curves, let's stick to the dotted flow)
        
        return image

        return image
    
    def draw_pingala_flows(self, image):
        """Draws connections between limbs using golden-ratio curves."""
        # Key aesthetic: Everything flows from the Center (Torso) outwards, 
        # or abstract relationships between extremities.
        
        white = (200, 200, 200)
        
        # Center of gravity approx (Mid Hips)
        p_center = (int((self.get_pt(23)[0] + self.get_pt(24)[0])/2), 
                   int((self.get_pt(23)[1] + self.get_pt(24)[1])/2))
        
        # Radiating curves to extremities
        extremities = [
            15, # L Wrist
            16, # R Wrist
            27, # L Ankle
            28, # R Ankle
            0,  # Nose
        ]

        # Elbows and Knees - important for dance dynamics
        intermediate_joints = [
            13, # L Elbow
            14, # R Elbow
            25, # L Knee
            26, # R Knee
        ]
        
        # Bounding boxes for Elbows/Knees (Small, focused)
        self.draw_bounding_box(image, [13], label_prefix="L-Elbow")
        self.draw_bounding_box(image, [14], label_prefix="R-Elbow")
        self.draw_bounding_box(image, [25], label_prefix="L-Knee")
        self.draw_bounding_box(image, [26], label_prefix="R-Knee")

        # 1. Torso -> Elbow -> Wrist (Arm Flow)
        # We need "Joint Flow". 
        # L Shoulder(11) -> L Elbow(13) -> L Wrist(15)
        self.draw_parabola_curve(image, self.get_pt(11), self.get_pt(13), white, 1, dashed=True)
        self.draw_parabola_curve(image, self.get_pt(13), self.get_pt(15), white, 1, dashed=True)
        
        # R Shoulder(12) -> R Elbow(14) -> R Wrist(16)
        self.draw_parabola_curve(image, self.get_pt(12), self.get_pt(14), white, 1, dashed=True)
        self.draw_parabola_curve(image, self.get_pt(14), self.get_pt(16), white, 1, dashed=True)
        
        # 2. Hip -> Knee -> Ankle (Leg Flow)
        # L Hip(23) -> L Knee(25) -> L Ankle(27)
        self.draw_parabola_curve(image, self.get_pt(23), self.get_pt(25), white, 1, dashed=True)
        self.draw_parabola_curve(image, self.get_pt(25), self.get_pt(27), white, 1, dashed=True)
        
        # R Hip(24) -> R Knee(26) -> R Ankle(28)
        self.draw_parabola_curve(image, self.get_pt(24), self.get_pt(26), white, 1, dashed=True)
        self.draw_parabola_curve(image, self.get_pt(26), self.get_pt(28), white, 1, dashed=True)

        
        # Radiating abstract curves (Center to Extremities) - kept for "Sci-Fi" look
        for idx in extremities:
            p_ext = self.get_pt(idx)
            # Draw Golden Curve
            self.draw_parabola_curve(image, p_center, p_ext, white, 1, dashed=True)
            
        
        # Inter-limb connections (The "Web")
        # Hand to Hand
        self.draw_parabola_curve(image, self.get_pt(15), self.get_pt(16), (255, 255, 255), 1, dashed=True)
        # Foot to Foot
        self.draw_parabola_curve(image, self.get_pt(27), self.get_pt(28), (255, 255, 255), 1, dashed=True)
        
        # New: Elbow to Elbow / Knee to Knee (Structure)
        self.draw_parabola_curve(image, self.get_pt(13), self.get_pt(14), (200, 200, 255), 1, dashed=True)
        self.draw_parabola_curve(image, self.get_pt(25), self.get_pt(26), (200, 200, 255), 1, dashed=True)
        
        # Cross Connect (Hand to Opposite Foot) - Very Vitruvian Man
        self.draw_parabola_curve(image, self.get_pt(15), self.get_pt(28), (255, 255, 200), 1, dashed=True)
        self.draw_parabola_curve(image, self.get_pt(16), self.get_pt(27), (255, 255, 200), 1, dashed=True)

        return image
