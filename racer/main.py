import cv2
import numpy as np
import pygame
import sys
import mediapipe as mp
import math
import random
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Game Constants
WIDTH, HEIGHT = 800, 600
ROAD_COLOR = (50, 50, 50)
GRASS_COLOR = (0, 100, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
GREEN = (0, 255, 0)
LIGHT_BLUE = (100, 100, 255)  # Added missing LIGHT_BLUE color

# Game settings
FPS = 60
ROAD_WIDTH = 400
LANE_WIDTH = ROAD_WIDTH // 3
CAR_WIDTH = 40
CAR_HEIGHT = 70
OBSTACLE_WIDTH = 50
OBSTACLE_HEIGHT = 50
BASE_SPEED = 5
MAX_SPEED = 20
ACCELERATION = 0.2
DECELERATION = 0.1

# Create game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Perfect Hand Gesture Racing Game")
clock = pygame.time.Clock()

# Load sounds
try:
    engine_sound = pygame.mixer.Sound('engine.wav')
    crash_sound = pygame.mixer.Sound('crash.wav')
    score_sound = pygame.mixer.Sound('score.wav')
except:
    # Create silent sounds if files not found
    engine_sound = pygame.mixer.Sound(buffer=bytearray([]))
    crash_sound = pygame.mixer.Sound(buffer=bytearray([]))
    score_sound = pygame.mixer.Sound(buffer=bytearray([]))

class PlayerCar:
    def __init__(self):
        self.width = CAR_WIDTH
        self.height = CAR_HEIGHT
        self.x = WIDTH // 2
        self.y = HEIGHT - 100
        self.speed = BASE_SPEED
        self.lane = 1  # 0: left, 1: center, 2: right
        self.lanes = [WIDTH//2 - LANE_WIDTH, WIDTH//2, WIDTH//2 + LANE_WIDTH]
        self.target_x = self.lanes[self.lane]
        self.color = RED
        self.health = 100
        self.score = 0
        self.drift_offset = 0
        self.smoke_particles = []
        self.boost_active = False
        self.boost_timer = 0
        
    def update(self, steering, acceleration):
        # Handle lane changes with steering input
        if steering == "left" and self.lane > 0:
            self.lane -= 1
            self.target_x = self.lanes[self.lane]
        elif steering == "right" and self.lane < 2:
            self.lane += 1
            self.target_x = self.lanes[self.lane]
            
        # Smooth movement to target lane
        dx = self.target_x - self.x
        self.x += dx * 0.3  # Faster lane changes
        
        # Add drift effect based on steering intensity
        self.drift_offset = dx * 0.15
        
        # Update speed based on acceleration input
        if acceleration == "accelerate":
            self.speed = min(MAX_SPEED, self.speed + ACCELERATION)
            self.boost_active = True
            self.boost_timer = 10
        elif acceleration == "brake":
            self.speed = max(BASE_SPEED, self.speed - DECELERATION)
            self.boost_active = False
        else:
            # Gradually return to base speed
            if self.speed > BASE_SPEED:
                self.speed -= DECELERATION * 0.5
            self.boost_active = False
            
        # Update boost timer
        if self.boost_timer > 0:
            self.boost_timer -= 1
            
        # Add smoke/boost particles
        if self.speed > BASE_SPEED + 3 and random.random() < 0.4:
            color = (255, 100, 0) if self.boost_active else (100, 100, 100)
            self.smoke_particles.append({
                'x': self.x + random.randint(-15, 15),
                'y': self.y + self.height//2,
                'size': random.randint(8, 20),
                'life': 40,
                'color': color
            })
            
        # Update particles
        for particle in self.smoke_particles[:]:
            particle['y'] -= 3
            particle['life'] -= 1
            if particle['life'] <= 0:
                self.smoke_particles.remove(particle)
        
    def draw(self):
        # Draw smoke/boost particles
        for particle in self.smoke_particles:
            alpha = min(255, particle['life'] * 6)
            smoke_surface = pygame.Surface((particle['size'], particle['size']), pygame.SRCALPHA)
            pygame.draw.circle(smoke_surface, (*particle['color'], alpha), 
                             (particle['size']//2, particle['size']//2), particle['size']//2)
            screen.blit(smoke_surface, (particle['x'], particle['y']))
        
        # Draw car body with drift effect
        car_rect = pygame.Rect(
            self.x - self.width//2 + self.drift_offset, 
            self.y - self.height//2, 
            self.width, 
            self.height
        )
        
        # Car body with boost effect
        car_color = (255, 200, 0) if self.boost_active else self.color
        pygame.draw.rect(screen, car_color, car_rect)
        pygame.draw.rect(screen, BLACK, car_rect, 2)
        
        # Draw racing stripes when boosting
        if self.boost_active:
            stripe_rect = pygame.Rect(
                self.x - self.width//2 + self.drift_offset,
                self.y - self.height//2 + 5,
                self.width,
                5
            )
            pygame.draw.rect(screen, WHITE, stripe_rect)
        
        # Draw windows
        window_rect = pygame.Rect(
            self.x - self.width//2 + 5 + self.drift_offset,
            self.y - self.height//2 + 5,
            self.width - 10,
            15
        )
        pygame.draw.rect(screen, (150, 200, 255), window_rect)
        
        # Draw headlights
        headlight_color = (255, 255, 200) if self.boost_active else YELLOW
        pygame.draw.circle(screen, headlight_color, 
                         (int(self.x - 10 + self.drift_offset), int(self.y + self.height//2 - 5)), 6)
        pygame.draw.circle(screen, headlight_color, 
                         (int(self.x + 10 + self.drift_offset), int(self.y + self.height//2 - 5)), 6)
        
        # Draw wheels with drift effect
        wheel_angle = math.radians(20 if abs(self.drift_offset) > 2 else 0)
        self.draw_wheel(self.x - self.width//2 - 3, self.y - self.height//2 + 10, wheel_angle)
        self.draw_wheel(self.x + self.width//2 - 3, self.y - self.height//2 + 10, -wheel_angle)
        self.draw_wheel(self.x - self.width//2 - 3, self.y + self.height//2 - 25, wheel_angle)
        self.draw_wheel(self.x + self.width//2 - 3, self.y + self.height//2 - 25, -wheel_angle)
    
    def draw_wheel(self, x, y, angle):
        # Draw rotated wheels for drift effect
        wheel_surface = pygame.Surface((6, 15), pygame.SRCALPHA)
        pygame.draw.rect(wheel_surface, BLACK, (0, 0, 6, 15))
        rotated_wheel = pygame.transform.rotate(wheel_surface, math.degrees(angle))
        screen.blit(rotated_wheel, (x + self.drift_offset - rotated_wheel.get_width()//2, 
                                  y - rotated_wheel.get_height()//2))

class Obstacle:
    def __init__(self, lane, speed_multiplier=1.0):
        self.width = OBSTACLE_WIDTH
        self.height = OBSTACLE_HEIGHT
        self.lanes = [WIDTH//2 - LANE_WIDTH, WIDTH//2, WIDTH//2 + LANE_WIDTH]
        self.x = self.lanes[lane]
        self.y = -OBSTACLE_HEIGHT
        self.lane = lane
        self.speed = BASE_SPEED * speed_multiplier
        self.colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0)]
        self.color = random.choice(self.colors)
        self.type = random.choice(["car", "truck", "cone", "barrier"])
        
    def update(self, player_speed):
        self.y += player_speed
        
    def draw(self):
        if self.type == "car":
            pygame.draw.rect(screen, self.color, 
                           (self.x - self.width//2, self.y, self.width, self.height))
            pygame.draw.rect(screen, BLACK, 
                           (self.x - self.width//2, self.y, self.width, self.height), 2)
            # Windows
            pygame.draw.rect(screen, (150, 200, 255), 
                           (self.x - self.width//2 + 5, self.y + 5, self.width - 10, 10))
        elif self.type == "truck":
            pygame.draw.rect(screen, self.color, 
                           (self.x - self.width//2, self.y, self.width, self.height + 10))
            pygame.draw.rect(screen, BLACK, 
                           (self.x - self.width//2, self.y, self.width, self.height + 10), 2)
        elif self.type == "cone":
            points = [
                (self.x, self.y),
                (self.x - self.width//2, self.y + self.height),
                (self.x + self.width//2, self.y + self.height)
            ]
            pygame.draw.polygon(screen, ORANGE, points)
            pygame.draw.polygon(screen, BLACK, points, 2)
        else:  # barrier
            pygame.draw.rect(screen, (100, 100, 100), 
                           (self.x - self.width//2, self.y, self.width, self.height//2))
            pygame.draw.rect(screen, RED, 
                           (self.x - self.width//2, self.y, self.width, self.height//2), 2)
        
    def collides_with(self, player):
        player_rect = pygame.Rect(
            player.x - player.width//2,
            player.y - player.height//2,
            player.width,
            player.height
        )
        obstacle_rect = pygame.Rect(
            self.x - self.width//2,
            self.y,
            self.width,
            self.height
        )
        return player_rect.colliderect(obstacle_rect)

class RacingGame:
    def __init__(self):
        self.player = PlayerCar()
        self.obstacles = []
        self.score = 0
        self.game_over = False
        self.font = pygame.font.SysFont('Arial', 24)
        self.big_font = pygame.font.SysFont('Arial', 48)
        self.obstacle_timer = 0
        self.road_scroll = 0
        self.difficulty_level = 1
        self.background_items = []
        self.generate_background()
        
    def generate_background(self):
        self.background_items = []
        for i in range(15):
            self.background_items.append({
                'x': random.randint(0, WIDTH),
                'y': random.randint(-HEIGHT, 0),
                'type': random.choice(['tree', 'building', 'billboard']),
                'speed': random.uniform(0.5, 2.0),
                'size': random.randint(30, 60)
            })
        
    def update(self, steering, acceleration):
        if self.game_over:
            return
            
        # Update player with both steering and acceleration
        self.player.update(steering, acceleration)
        
        # Update road scrolling based on player speed
        self.road_scroll += self.player.speed
        if self.road_scroll >= 50:
            self.road_scroll = 0
            
        # Update background
        for item in self.background_items:
            item['y'] += self.player.speed * item['speed']
            if item['y'] > HEIGHT:
                item['y'] = -100
                item['x'] = random.randint(0, WIDTH)
            
        # Increase difficulty with score
        self.difficulty_level = 1 + (self.score // 10) * 0.2
        
        # Spawn obstacles
        self.obstacle_timer += 1
        spawn_rate = max(20, 60 - (self.score // 2))
        if self.obstacle_timer > spawn_rate:
            lane = random.randint(0, 2)
            speed_multiplier = random.uniform(1.0, 1.5) * self.difficulty_level
            self.obstacles.append(Obstacle(lane, speed_multiplier))
            self.obstacle_timer = 0
            
        # Update obstacles and check collisions
        for obstacle in self.obstacles[:]:
            obstacle.update(self.player.speed)
            
            if obstacle.collides_with(self.player):
                damage = 25 if obstacle.type == "barrier" else 15
                self.player.health -= damage
                crash_sound.play()
                
                # Add collision effect
                for _ in range(10):
                    self.player.smoke_particles.append({
                        'x': self.player.x + random.randint(-20, 20),
                        'y': self.player.y + random.randint(-10, 10),
                        'size': random.randint(5, 15),
                        'life': 20,
                        'color': (255, 50, 50)
                    })
                
                if self.player.health <= 0:
                    self.game_over = True
                self.obstacles.remove(obstacle)
            elif obstacle.y > HEIGHT:
                self.obstacles.remove(obstacle)
                self.score += 1
                if self.score % 5 == 0:
                    score_sound.play()
        
    def draw(self):
        # Draw sky with gradient
        for y in range(HEIGHT):
            sky_color = (135, 206, 235) if y < HEIGHT//2 else (100, 180, 255)
            pygame.draw.line(screen, sky_color, (0, y), (WIDTH, y))
        
        # Draw background items
        for item in self.background_items:
            if item['type'] == 'tree':
                pygame.draw.rect(screen, (101, 67, 33), 
                               (item['x'], item['y'], 10, item['size']))
                pygame.draw.circle(screen, (0, 80, 0), 
                                 (item['x'] + 5, item['y']), item['size'] // 2)
            elif item['type'] == 'building':
                pygame.draw.rect(screen, (70, 70, 70), 
                               (item['x'], item['y'], item['size'], item['size'] * 2))
                for i in range(3):
                    for j in range(4):
                        pygame.draw.rect(screen, (255, 255, 100), 
                                       (item['x'] + 5 + j * 10, item['y'] + 5 + i * 15, 8, 10))
            else:  # billboard
                pygame.draw.rect(screen, (200, 200, 200), 
                               (item['x'], item['y'], item['size'], item['size']//2))
                pygame.draw.rect(screen, RED, 
                               (item['x'], item['y'], item['size'], item['size']//2), 2)
        
        # Draw grass
        pygame.draw.rect(screen, GRASS_COLOR, (0, 0, (WIDTH - ROAD_WIDTH) // 2, HEIGHT))
        pygame.draw.rect(screen, GRASS_COLOR, ((WIDTH + ROAD_WIDTH) // 2, 0, (WIDTH - ROAD_WIDTH) // 2, HEIGHT))
        
        # Draw road with perspective
        road_rect = pygame.Rect((WIDTH - ROAD_WIDTH) // 2, 0, ROAD_WIDTH, HEIGHT)
        pygame.draw.rect(screen, ROAD_COLOR, road_rect)
        
        # Draw road markings with perspective
        for i in range(-1, HEIGHT // 50 + 1):
            line_y = (i * 50 + self.road_scroll) % (HEIGHT + 50)
            line_width = 10 - (line_y / HEIGHT) * 5  # Perspective effect
            pygame.draw.rect(screen, WHITE, 
                           (WIDTH // 2 - line_width//2, line_y, line_width, 30))
            
        # Draw lane dividers
        pygame.draw.rect(screen, WHITE, 
                       (WIDTH // 2 - LANE_WIDTH - 2, 0, 4, HEIGHT))
        pygame.draw.rect(screen, WHITE, 
                       (WIDTH // 2 + LANE_WIDTH - 2, 0, 4, HEIGHT))
        
        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle.draw()
            
        # Draw player
        self.player.draw()
        
        # Draw HUD with transparent background
        hud_bg = pygame.Surface((WIDTH, 80), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 128))
        screen.blit(hud_bg, (0, 0))
        
        # Draw speedometer with color coding
        speed_color = GREEN if self.player.speed < BASE_SPEED + 5 else YELLOW if self.player.speed < MAX_SPEED - 3 else RED
        speed_text = self.font.render(f"SPEED: {int(self.player.speed * 15)} KM/H", True, speed_color)
        screen.blit(speed_text, (20, 20))
        
        # Draw score
        score_text = self.font.render(f"SCORE: {self.score}", True, WHITE)
        screen.blit(score_text, (20, 50))
        
        # Draw health bar with animation
        health_text = self.font.render("HEALTH:", True, WHITE)
        screen.blit(health_text, (WIDTH - 200, 20))
        
        health_bg = pygame.Rect(WIDTH - 120, 20, 100, 20)
        health_fg = pygame.Rect(WIDTH - 120, 20, 100 * (self.player.health / 100), 20)
        
        health_color = GREEN if self.player.health > 60 else YELLOW if self.player.health > 30 else RED
        pygame.draw.rect(screen, (50, 50, 50), health_bg)
        pygame.draw.rect(screen, health_color, health_fg)
        pygame.draw.rect(screen, WHITE, health_bg, 2)
        
        # Draw boost indicator
        if self.player.boost_active:
            boost_text = self.font.render("BOOST!", True, ORANGE)
            screen.blit(boost_text, (WIDTH - 200, 50))
        
        # Draw difficulty indicator
        diff_text = self.font.render(f"LEVEL: {int(self.difficulty_level)}", True, YELLOW)
        screen.blit(diff_text, (WIDTH // 2 - diff_text.get_width() // 2, 20))
        
        # Draw controls hint
        controls_text = self.font.render("HAND GESTURES: TILT TO STEER | THUMB=GAS | PINKY=BRAKE", True, LIGHT_BLUE)
        screen.blit(controls_text, (WIDTH // 2 - controls_text.get_width() // 2, HEIGHT - 30))
        
        # Draw game over screen
        if self.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            screen.blit(overlay, (0, 0))
            
            game_over_text = self.big_font.render("RACE OVER", True, RED)
            score_text = self.font.render(f"FINAL SCORE: {self.score}", True, WHITE)
            restart_text = self.font.render("PRESS R TO RESTART OR Q TO QUIT", True, YELLOW)
            
            screen.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, HEIGHT//2 - 60))
            screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT//2))
            screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT//2 + 50))
        
        pygame.display.flip()
        
    def reset(self):
        self.__init__()

def detect_steering_direction(hand_landmarks):
    """
    Perfect steering detection using multiple reference points
    """
    landmarks = hand_landmarks.landmark
    
    # Method 1: Palm orientation (primary)
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    dx_palm = middle_mcp.x - wrist.x
    
    # Method 2: Finger spread direction (secondary)
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    dx_fingers = pinky_mcp.x - index_mcp.x
    
    # Method 3: Hand rotation (tertiary)
    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    wrist_rotation = thumb_cmc.x - wrist.x
    
    # Combine methods with weights
    dx = dx_palm * 0.6 + dx_fingers * 0.3 + wrist_rotation * 0.1
    
    # Adaptive dead zone based on hand size
    hand_size = math.sqrt((middle_mcp.x - wrist.x)**2 + (middle_mcp.y - wrist.y)**2)
    dead_zone = 0.06 + (hand_size * 0.04)  # Larger hands get larger dead zone
    
    # Determine steering direction
    if abs(dx) > dead_zone:
        if dx > 0:
            return "right"
        else:
            return "left"
    
    return "center"

def detect_acceleration_gesture(hand_landmarks):
    """
    Improved acceleration/brake detection with multiple checks
    """
    landmarks = hand_landmarks.landmark
    
    # Thumb detection for acceleration
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    
    # Pinky detection for brake
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    
    # Calculate angles and distances for better detection
    thumb_angle = math.degrees(math.atan2(thumb_tip.y - thumb_ip.y, thumb_tip.x - thumb_ip.x))
    pinky_angle = math.degrees(math.atan2(pinky_tip.y - pinky_dip.y, pinky_tip.x - pinky_dip.x))
    
    # Thumb up detection (accelerate)
    thumb_up = (thumb_tip.y < thumb_ip.y and thumb_tip.y < thumb_mcp.y and 
                abs(thumb_angle) > 45)
    
    # Pinky up detection (brake)
    pinky_up = (pinky_tip.y < pinky_dip.y and pinky_tip.y < pinky_mcp.y and 
                abs(pinky_angle) > 30)
    
    if thumb_up and not pinky_up:
        return "accelerate"
    elif pinky_up and not thumb_up:
        return "brake"
    
    return "maintain"

def draw_advanced_gesture_info(frame, steering, acceleration, confidence, landmarks=None):
    """Draw comprehensive gesture information"""
    h, w = frame.shape[:2]
    
    # Draw steering information with confidence
    steering_text = f"STEERING: {steering.upper()}"
    confidence_color = (0, 255, 0) if confidence > 0.7 else (255, 255, 0) if confidence > 0.4 else (255, 0, 0)
    cv2.putText(frame, steering_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, confidence_color, 2)
    
    # Draw acceleration information
    accel_color = (0, 255, 0) if acceleration == "accelerate" else (255, 0, 0) if acceleration == "brake" else (255, 255, 0)
    accel_text = f"PEDAL: {acceleration.upper()}"
    cv2.putText(frame, accel_text, (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, accel_color, 2)
    
    # Draw control guide
    guide = [
        "PERFECT RACING CONTROLS:",
        "TILT HAND ← → FOR STEERING",
        "THUMB ↑ = ACCELERATE",
        "PINKY ↑ = BRAKE",
        "KEEP HAND STEADY = CRUISE"
    ]
    
    for i, text in enumerate(guide):
        cv2.putText(frame, text, (w - 400, 40 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Draw steering wheel visualization
    center_x, center_y = w // 2, h - 80
    wheel_radius = 40
    
    # Draw steering wheel base
    cv2.circle(frame, (center_x, center_y), wheel_radius, (100, 100, 100), 3)
    
    # Draw steering indicator
    if steering == "left":
        angle = math.radians(-30)
    elif steering == "right":
        angle = math.radians(30)
    else:
        angle = 0
        
    end_x = int(center_x + wheel_radius * math.sin(angle))
    end_y = int(center_y - wheel_radius * math.cos(angle))
    
    cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 4)
    
    # Draw pedal indicators
    gas_x, gas_y = center_x - 60, center_y + 50
    brake_x, brake_y = center_x + 60, center_y + 50
    
    gas_color = (0, 255, 0) if acceleration == "accelerate" else (50, 50, 50)
    brake_color = (255, 0, 0) if acceleration == "brake" else (50, 50, 50)
    
    cv2.rectangle(frame, (gas_x-20, gas_y-10), (gas_x+20, gas_y+10), gas_color, -1)
    cv2.rectangle(frame, (brake_x-20, brake_y-10), (brake_x+20, brake_y+10), brake_color, -1)
    
    cv2.putText(frame, "GAS", (gas_x-15, gas_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "BRAKE", (brake_x-20, brake_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw hand landmarks with connections
    if landmarks:
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

def main():
    game = RacingGame()
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("PERFECT HAND GESTURE RACING GAME!")
    print("=" * 50)
    print("CONTROLS:")
    print("- Tilt hand LEFT/RIGHT to steer")
    print("- Thumb UP to ACCELERATE") 
    print("- Pinky UP to BRAKE")
    print("- Keep hand steady to maintain speed")
    print("- Avoid obstacles and score points!")
    print("=" * 50)
    print("Press 'Q' in camera window to quit")
    print("Press 'R' to restart after game over")
    
    # Gesture tracking variables
    steering_history = deque(maxlen=5)
    acceleration_history = deque(maxlen=5)
    last_steering = "center"
    last_acceleration = "maintain"
    
    # Play engine sound
    engine_sound.play(-1)
    engine_sound.set_volume(0.3)
    
    while True:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                engine_sound.stop()
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game.game_over:
                    game.reset()
                    engine_sound.play(-1)
                    print("Game reset!")
                elif event.key == pygame.K_q:
                    engine_sound.stop()
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
        
        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            continue
            
        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        current_steering = "center"
        current_acceleration = "maintain"
        gesture_confidence = 0.5
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Detect gestures
                steering = detect_steering_direction(hand_landmarks)
                acceleration = detect_acceleration_gesture(hand_landmarks)
                
                # Add to history for smoothing
                steering_history.append(steering)
                acceleration_history.append(acceleration)
                
                # Get most common gesture from history (majority vote)
                if steering_history:
                    current_steering = max(set(steering_history), key=steering_history.count)
                if acceleration_history:
                    current_acceleration = max(set(acceleration_history), key=acceleration_history.count)
                
                # Calculate confidence based on consistency
                steering_consistency = steering_history.count(current_steering) / len(steering_history)
                accel_consistency = acceleration_history.count(current_acceleration) / len(acceleration_history)
                gesture_confidence = (steering_consistency + accel_consistency) / 2
                
                # Update game with smoothed gestures
                game.update(current_steering, current_acceleration)
                
                last_steering = current_steering
                last_acceleration = current_acceleration
        
        # Draw advanced gesture information
        draw_advanced_gesture_info(frame, current_steering, current_acceleration, 
                                 gesture_confidence, results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None)
        
        # Display camera feed
        cv2.imshow('PERFECT HAND GESTURE RACING - Press Q to quit', frame)
        cv2.resizeWindow('PERFECT HAND GESTURE RACING - Press Q to quit', 640, 480)
        
        # Draw game
        game.draw()
        
        # Control frame rate
        clock.tick(FPS)
        
        # Exit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    engine_sound.stop()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()