import cv2
import numpy as np
import pygame
import sys
import mediapipe as mp
import random
import math
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 1024, 768
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
DARK_BLUE = (25, 25, 50)

# Fixed Background Colors - Ensure proper length
BG_COLORS = [
    (25, 42, 86), 
    (31, 58, 147), 
    (44, 62, 80)
]

# Blade colors
BLADE_COLORS = [
    (57, 255, 20),   # Neon Green
    (0, 255, 255),   # Cyan
    (255, 20, 147),  # Neon Pink
]

# Fruit types
FRUITS = [
    {"name": "apple", "color": (255, 50, 50), "points": 10, "radius": 35},
    {"name": "banana", "color": (255, 255, 100), "points": 15, "radius": 40},
    {"name": "orange", "color": (255, 150, 0), "points": 20, "radius": 32},
    {"name": "watermelon", "color": (0, 180, 0), "points": 25, "radius": 45},
    {"name": "strawberry", "color": (255, 50, 100), "points": 30, "radius": 28},
]

# Bomb
BOMB = {"color": (40, 40, 40), "radius": 40, "explosion_radius": 200}

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.radius = random.uniform(2, 6)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = 1.0
        self.decay = random.uniform(0.02, 0.05)
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1
        self.life -= self.decay
        return self.life > 0
        
    def draw(self, screen):
        alpha = int(self.life * 255)
        surf = pygame.Surface((int(self.radius * 4), int(self.radius * 4)), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*self.color, alpha), 
                          (int(self.radius * 2), int(self.radius * 2)), int(self.radius))
        screen.blit(surf, (int(self.x - self.radius * 2), int(self.y - self.radius * 2)))

class Fruit:
    def __init__(self):
        self.type = random.choice(FRUITS)
        self.x = random.randint(150, WIDTH - 150)
        self.y = HEIGHT + 50
        angle = random.uniform(math.pi/4, 3*math.pi/4)
        speed = random.uniform(18, 25)
        self.vx = math.cos(angle) * speed * random.choice([-1, 1])
        self.vy = -speed
        self.rotation = 0
        self.rotation_speed = random.uniform(-8, 8)
        self.sliced = False
        self.sliced_time = 0
        self.particles = []
        self.missed = False
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.3
        self.rotation += self.rotation_speed
        
        # Update particles
        self.particles = [p for p in self.particles if p.update()]
        
        # Add trail particles
        if not self.sliced and random.random() < 0.2:
            self.particles.append(Particle(
                self.x + random.uniform(-15, 15),
                self.y + random.uniform(-15, 15),
                self.type["color"]
            ))
        
        # Check if off screen
        if self.y > HEIGHT + 100:
            self.missed = True
            
        return not self.missed
        
    def draw(self, screen):
        # Draw particles
        for particle in self.particles:
            particle.draw(screen)
            
        if not self.sliced:
            # Draw fruit
            size = self.type["radius"] * 2
            fruit_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            
            # Main fruit
            pygame.draw.circle(fruit_surf, self.type["color"], 
                             (self.type["radius"], self.type["radius"]), 
                             self.type["radius"])
            
            # Highlight
            highlight_size = self.type["radius"] // 3
            highlight_pos = (self.type["radius"] - highlight_size, self.type["radius"] - highlight_size)
            pygame.draw.circle(fruit_surf, (255, 255, 255, 180), highlight_pos, highlight_size)
            
            # Rotate and draw
            rotated = pygame.transform.rotate(fruit_surf, self.rotation)
            screen.blit(rotated, 
                       (self.x - rotated.get_width() // 2, 
                        self.y - rotated.get_height() // 2))
        else:
            # Draw sliced halves
            time_since_sliced = pygame.time.get_ticks() - self.sliced_time
            offset = min(time_since_sliced * 0.2, 30)
            
            # Left half
            pygame.draw.circle(screen, self.type["color"], 
                             (int(self.x - offset), int(self.y)), 
                             self.type["radius"] // 2)
            # Right half  
            pygame.draw.circle(screen, self.type["color"],
                             (int(self.x + offset), int(self.y)),
                             self.type["radius"] // 2)
            
    def slice(self):
        if not self.sliced:
            self.sliced = True
            self.sliced_time = pygame.time.get_ticks()
            
            # Create slice particles
            for _ in range(15):
                self.particles.append(Particle(self.x, self.y, self.type["color"]))
                
            return self.type["points"]
        return 0

class Bomb:
    def __init__(self):
        self.x = random.randint(150, WIDTH - 150)
        self.y = HEIGHT + 50
        angle = random.uniform(math.pi/4, 3*math.pi/4)
        speed = random.uniform(12, 18)
        self.vx = math.cos(angle) * speed * random.choice([-1, 1])
        self.vy = -speed
        self.rotation = 0
        self.rotation_speed = random.uniform(-5, 5)
        self.exploded = False
        self.explosion_time = 0
        self.particles = []
        self.missed = False
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.25
        self.rotation += self.rotation_speed
        
        # Update particles
        self.particles = [p for p in self.particles if p.update()]
        
        if self.y > HEIGHT + 100:
            self.missed = True
            
        return not self.exploded and not self.missed
        
    def draw(self, screen):
        # Draw particles
        for particle in self.particles:
            particle.draw(screen)
            
        if not self.exploded:
            # Draw bomb
            size = BOMB["radius"] * 2
            bomb_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            
            # Bomb body
            pygame.draw.circle(bomb_surf, BOMB["color"],
                             (BOMB["radius"], BOMB["radius"]), BOMB["radius"])
            
            # Fuse
            fuse_points = [
                (BOMB["radius"] + 15, BOMB["radius"] - 10),
                (BOMB["radius"] + 25, BOMB["radius"] - 15),
                (BOMB["radius"] + 20, BOMB["radius"] - 25)
            ]
            pygame.draw.lines(bomb_surf, (139, 69, 19), False, fuse_points, 3)
            
            # Danger stripes
            for i in range(4):
                start_angle = i * math.pi / 2
                pygame.draw.arc(bomb_surf, (255, 0, 0),
                              [5, 5, size - 10, size - 10],
                              start_angle, start_angle + math.pi / 4, 3)
            
            # Rotate and draw
            rotated = pygame.transform.rotate(bomb_surf, self.rotation)
            screen.blit(rotated,
                       (self.x - rotated.get_width() // 2,
                        self.y - rotated.get_height() // 2))
        else:
            # Draw explosion
            time_since_explosion = pygame.time.get_ticks() - self.explosion_time
            if time_since_explosion < 1000:
                radius = min(BOMB["explosion_radius"], time_since_explosion * 0.3)
                pulse = (math.sin(time_since_explosion * 0.02) + 1) * 0.2 + 0.8
                current_radius = int(radius * pulse)
                
                explosion_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
                for r in range(current_radius, 0, -5):
                    alpha = int(255 * (r / current_radius))
                    pygame.draw.circle(explosion_surf, (255, 100, 0, alpha),
                                     (current_radius, current_radius), r)
                
                screen.blit(explosion_surf,
                           (self.x - current_radius, self.y - current_radius))
                
    def explode(self):
        if not self.exploded:
            self.exploded = True
            self.explosion_time = pygame.time.get_ticks()
            
            # Create explosion particles
            for _ in range(25):
                self.particles.append(Particle(self.x, self.y, 
                                             random.choice([(255, 100, 0), (255, 200, 0)])))
            return True
        return False

class Blade:
    def __init__(self):
        self.points = deque(maxlen=15)
        self.color = random.choice(BLADE_COLORS)
        
    def add_point(self, x, y):
        current_time = pygame.time.get_ticks()
        self.points.append((x, y, current_time))
        
    def update(self):
        # Remove old points
        current_time = pygame.time.get_ticks()
        self.points = deque([p for p in self.points if current_time - p[2] < 200], maxlen=15)
        
    def draw(self, screen):
        if len(self.points) > 1:
            # Draw blade trail
            for i in range(1, len(self.points)):
                progress = i / len(self.points)
                alpha = int(255 * progress)
                width = max(2, int(8 * progress))
                
                start_pos = self.points[i-1][:2]
                end_pos = self.points[i][:2]
                
                # Create surface for this segment
                segment_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(segment_surf, (*self.color, alpha), start_pos, end_pos, width)
                screen.blit(segment_surf, (0, 0))

class FruitNinjaGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Fruit Ninja - Hand Controlled")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.big_font = pygame.font.SysFont('arial', 72, bold=True)
        self.medium_font = pygame.font.SysFont('arial', 42, bold=True)
        self.small_font = pygame.font.SysFont('arial', 28)
        
        self.reset_game()
        
    def reset_game(self):
        self.fruits = []
        self.bombs = []
        self.score = 0
        self.lives = 3
        self.combo = 0
        self.combo_time = 0
        self.game_over = False
        self.last_fruit_time = 0
        self.last_bomb_time = 0
        self.blade = Blade()
        self.last_hand_pos = None
        self.screen_shake = 0
        
    def spawn_fruit(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_fruit_time > 800:
            self.fruits.append(Fruit())
            self.last_fruit_time = current_time
            
    def spawn_bomb(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.last_bomb_time > 5000:
            if random.random() < 0.15:
                self.bombs.append(Bomb())
                self.last_bomb_time = current_time
            
    def detect_swipe(self, current_hand_pos):
        if self.last_hand_pos is None:
            self.last_hand_pos = current_hand_pos
            return None
            
        dx = current_hand_pos[0] - self.last_hand_pos[0]
        dy = current_hand_pos[1] - self.last_hand_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Enhanced detection
        if distance > 10:  # Reduced threshold for better sensitivity
            speed = distance * FPS
            if speed > 200:  # Lower speed threshold
                self.last_hand_pos = current_hand_pos
                return (dx, dy)
            
        self.last_hand_pos = current_hand_pos
        return None
        
    def check_slice(self, point1, point2):
        """Check if line between two points slices any fruits or bombs"""
        line_start = np.array(point1)
        line_end = np.array(point2)
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return
            
        line_unit = line_vec / line_len
        
        # Check fruits
        sliced_fruits = 0
        for fruit in self.fruits[:]:
            if not fruit.sliced:
                fruit_pos = np.array([fruit.x, fruit.y])
                point_vec = fruit_pos - line_start
                projection = np.dot(point_vec, line_unit)
                
                if 0 <= projection <= line_len:
                    closest_point = line_start + line_unit * projection
                    distance = np.linalg.norm(fruit_pos - closest_point)
                    
                    if distance < fruit.type["radius"]:
                        points = fruit.slice()
                        if points > 0:
                            self.score += points
                            sliced_fruits += 1
        
        # Update combo
        if sliced_fruits > 0:
            self.combo += sliced_fruits
            self.combo_time = pygame.time.get_ticks()
        
        # Check bombs
        for bomb in self.bombs[:]:
            if not bomb.exploded:
                bomb_pos = np.array([bomb.x, bomb.y])
                point_vec = bomb_pos - line_start
                projection = np.dot(point_vec, line_unit)
                
                if 0 <= projection <= line_len:
                    closest_point = line_start + line_unit * projection
                    distance = np.linalg.norm(bomb_pos - closest_point)
                    
                    if distance < BOMB["radius"]:
                        if bomb.explode():
                            self.lives -= 1
                            self.screen_shake = 25
                            self.combo = 0
                            if self.lives <= 0:
                                self.game_over = True
    
    def update(self, hand_pos=None):
        if self.game_over:
            return
            
        # Update combo timer
        current_time = pygame.time.get_ticks()
        if current_time - self.combo_time > 3000:
            self.combo = 0
            
        # Spawn objects
        self.spawn_fruit()
        self.spawn_bomb()
        
        # Update objects
        self.fruits = [f for f in self.fruits if f.update()]
        self.bombs = [b for b in self.bombs if b.update()]
        self.blade.update()
        
        # Remove exploded bombs
        self.bombs = [b for b in self.bombs if 
                     not b.exploded or 
                     pygame.time.get_ticks() - b.explosion_time < 1000]
        
        # Check for missed fruits
        missed_count = 0
        for fruit in self.fruits[:]:
            if fruit.missed and not fruit.sliced:
                self.fruits.remove(fruit)
                missed_count += 1
                
        if missed_count >= 3:
            self.lives -= 1
            self.screen_shake = 15
            if self.lives <= 0:
                self.game_over = True
        
        # Screen shake
        if self.screen_shake > 0:
            self.screen_shake -= 1
    
    def draw_background(self):
        """Fixed background drawing without index errors"""
        # Simple gradient without complex calculations
        for y in range(HEIGHT):
            # Simple vertical gradient
            t = y / HEIGHT
            if t < 0.33:
                color = BG_COLORS[0]
            elif t < 0.66:
                # Blend between first and second color
                blend = (t - 0.33) / 0.33
                r = int(BG_COLORS[0][0] * (1 - blend) + BG_COLORS[1][0] * blend)
                g = int(BG_COLORS[0][1] * (1 - blend) + BG_COLORS[1][1] * blend)
                b = int(BG_COLORS[0][2] * (1 - blend) + BG_COLORS[1][2] * blend)
                color = (r, g, b)
            else:
                # Blend between second and third color
                blend = (t - 0.66) / 0.34
                r = int(BG_COLORS[1][0] * (1 - blend) + BG_COLORS[2][0] * blend)
                g = int(BG_COLORS[1][1] * (1 - blend) + BG_COLORS[2][1] * blend)
                b = int(BG_COLORS[1][2] * (1 - blend) + BG_COLORS[2][2] * blend)
                color = (r, g, b)
            
            pygame.draw.line(self.screen, color, (0, y), (WIDTH, y))
    
    def draw(self):
        # Apply screen shake
        shake_x = random.randint(-self.screen_shake, self.screen_shake) if self.screen_shake > 0 else 0
        shake_y = random.randint(-self.screen_shake, self.screen_shake) if self.screen_shake > 0 else 0
        
        # Draw background
        self.screen.fill(DARK_BLUE)
        self.draw_background()
        
        # Draw game elements
        self.blade.draw(self.screen)
        
        for fruit in self.fruits:
            fruit.draw(self.screen)
        for bomb in self.bombs:
            bomb.draw(self.screen)
            
        # Draw UI
        # Score
        score_text = self.medium_font.render(f'SCORE: {self.score}', True, WHITE)
        self.screen.blit(score_text, (30, 30))
        
        # Lives
        lives_text = self.medium_font.render(f'LIVES: {self.lives}', True, WHITE)
        self.screen.blit(lives_text, (WIDTH - lives_text.get_width() - 30, 30))
        
        # Combo display
        if self.combo > 1:
            combo_color = GREEN if self.combo >= 5 else YELLOW
            combo_text = self.small_font.render(f'COMBO x{self.combo}!', True, combo_color)
            self.screen.blit(combo_text, (WIDTH // 2 - combo_text.get_width() // 2, 30))
        
        # Game over screen
        if self.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.big_font.render('GAME OVER', True, RED)
            final_score = self.medium_font.render(f'Final Score: {self.score}', True, WHITE)
            restart_text = self.small_font.render('Press R to Restart or Q to Quit', True, GREEN)
            
            self.screen.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, HEIGHT//2 - 80))
            self.screen.blit(final_score, (WIDTH//2 - final_score.get_width()//2, HEIGHT//2))
            self.screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT//2 + 60))
        
        pygame.display.flip()

def draw_hand_info(frame, hand_pos, swipe_detected, combo, score):
    h, w = frame.shape[:2]
    
    # Enhanced hand visualization
    if hand_pos:
        # Draw hand position
        cv2.circle(frame, (int(hand_pos[0]), int(hand_pos[1])), 12, (0, 255, 255), -1)
        cv2.circle(frame, (int(hand_pos[0]), int(hand_pos[1])), 15, (0, 200, 255), 2)
        
        status_color = (0, 255, 0) if swipe_detected else (255, 255, 0)
        status_text = "SLICING!" if swipe_detected else "Move hand to slice"
        cv2.putText(frame, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Game info
    cv2.putText(frame, f"SCORE: {score}", (w - 200, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if combo > 1:
        cv2.putText(frame, f"COMBO: x{combo}", (w - 200, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Instructions
    instructions = [
        "Move hand to slice fruits!",
        "Avoid black bombs!",
        "Build combos!",
        "R = Restart, Q = Quit"
    ]
    
    for i, text in enumerate(instructions):
        y_pos = 80 + i * 30
        cv2.putText(frame, text, (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    game = FruitNinjaGame()
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("🎯 FRUIT NINJA - PERFECT EDITION")
    print("🍎 Slice fruits with hand movements!")
    print("💣 Avoid bombs!")
    print("🎯 Build combos for high scores!")
    print("👋 Make sure your hand is visible in camera!")
    
    last_hand_pos = None
    
    while True:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset_game()
                    print("Game restarted!")
                elif event.key == pygame.K_q:
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
        
        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with hand detection
        results = hands.process(rgb_frame)
        
        current_hand_pos = None
        swipe_detected = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get index finger tip
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                current_hand_pos = (int(index_tip.x * w), int(index_tip.y * h))
                
                # Enhanced swipe detection
                swipe_data = game.detect_swipe(current_hand_pos)
                if swipe_data:
                    swipe_detected = True
                    
                    # Convert to game coordinates
                    game_x = int(current_hand_pos[0] * WIDTH / w)
                    game_y = int(current_hand_pos[1] * HEIGHT / h)
                    game.blade.add_point(game_x, game_y)
                    
                    # Check for slices
                    if last_hand_pos:
                        last_game_x = int(last_hand_pos[0] * WIDTH / w)
                        last_game_y = int(last_hand_pos[1] * HEIGHT / h)
                        game.check_slice((last_game_x, last_game_y), (game_x, game_y))
                
                last_hand_pos = current_hand_pos
        
        # Draw hand information
        draw_hand_info(frame, current_hand_pos, swipe_detected, game.combo, game.score)
        
        # Show camera feed
        cv2.imshow('Fruit Ninja - Hand Control (Press Q to quit)', frame)
        cv2.resizeWindow('Fruit Ninja - Hand Control (Press Q to quit)', 640, 480)
        
        # Update and draw game
        game.update(current_hand_pos)
        game.draw()
        
        # Maintain frame rate
        game.clock.tick(FPS)
        
        # Exit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()