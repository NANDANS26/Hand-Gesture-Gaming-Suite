import cv2
import numpy as np
import pygame
import sys
import mediapipe as mp
import random
from collections import deque
import math

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

# Game Constants
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 40
MAZE_WIDTH = WIDTH // CELL_SIZE
MAZE_HEIGHT = HEIGHT // CELL_SIZE
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
GRAY = (128, 128, 128)
LIGHT_BLUE = (173, 216, 230)
ORANGE = (255, 165, 0)

class MazeGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = None
    
    def generate_maze(self):
        # Initialize maze with walls
        self.maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        
        # Start from a random cell
        start_x, start_y = 1, 1
        self.maze[start_y][start_x] = 0
        
        # Use DFS to generate maze
        stack = [(start_x, start_y)]
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        
        while stack:
            current_x, current_y = stack[-1]
            random.shuffle(directions)
            
            found = False
            for dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                if 0 < nx < self.width - 1 and 0 < ny < self.height - 1 and self.maze[ny][nx] == 1:
                    # Remove wall between current and next cell
                    self.maze[current_y + dy//2][current_x + dx//2] = 0
                    self.maze[ny][nx] = 0
                    stack.append((nx, ny))
                    found = True
                    break
            
            if not found:
                stack.pop()
        
        # Ensure start and end are open
        self.maze[1][1] = 0  # Start
        self.maze[self.height-2][self.width-2] = 0  # End
        
        # Add some open spaces for easier navigation
        for _ in range(5):
            x, y = random.randint(2, self.width-3), random.randint(2, self.height-3)
            for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
                if 0 < x+dx < self.width-1 and 0 < y+dy < self.height-1:
                    self.maze[y+dy][x+dx] = 0
        
        return self.maze

class MazeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hand Gesture Controlled Maze Solver")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.big_font = pygame.font.SysFont('Arial', 48)
        
        self.maze_gen = MazeGenerator(MAZE_WIDTH, MAZE_HEIGHT)
        self.reset_game()
        
    def reset_game(self):
        self.maze = self.maze_gen.generate_maze()
        self.player_pos = [1, 1]  # Start position
        self.end_pos = [MAZE_WIDTH - 2, MAZE_HEIGHT - 2]  # End position
        self.game_won = False
        self.moves = 0
        self.start_time = pygame.time.get_ticks()
        self.player_trail = []  # Track player movement
        self.trail_length = 20
        
    def move_player(self, direction):
        if self.game_won:
            return
            
        dx, dy = direction
        new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
        
        # Check if move is valid (within bounds and not a wall)
        if (0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and 
            self.maze[new_y][new_x] == 0):
            self.player_pos = [new_x, new_y]
            self.moves += 1
            
            # Add to trail
            self.player_trail.append((new_x, new_y))
            if len(self.player_trail) > self.trail_length:
                self.player_trail.pop(0)
            
            # Check if reached end
            if self.player_pos == self.end_pos:
                self.game_won = True
                self.completion_time = (pygame.time.get_ticks() - self.start_time) // 1000
    
    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw maze
        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.maze[y][x] == 1:  # Wall
                    pygame.draw.rect(self.screen, BLUE, rect)
                    # Add texture to walls
                    pygame.draw.rect(self.screen, (0, 0, 150), rect, 1)
                else:  # Path
                    pygame.draw.rect(self.screen, BLACK, rect)
                    pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        
        # Draw player trail
        for i, (trail_x, trail_y) in enumerate(self.player_trail):
            alpha = int(255 * (i / len(self.player_trail)))
            trail_rect = pygame.Rect(
                trail_x * CELL_SIZE + 10, 
                trail_y * CELL_SIZE + 10, 
                CELL_SIZE - 20, CELL_SIZE - 20
            )
            pygame.draw.ellipse(self.screen, (255, 255, 0, alpha), trail_rect)
        
        # Draw start and end
        start_rect = pygame.Rect(1 * CELL_SIZE, 1 * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        end_rect = pygame.Rect(self.end_pos[0] * CELL_SIZE, self.end_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, GREEN, start_rect)
        pygame.draw.rect(self.screen, RED, end_rect)
        
        # Draw start/end labels
        start_text = self.font.render("START", True, WHITE)
        end_text = self.font.render("END", True, WHITE)
        self.screen.blit(start_text, (1 * CELL_SIZE + 5, 1 * CELL_SIZE + 10))
        self.screen.blit(end_text, (self.end_pos[0] * CELL_SIZE + 5, self.end_pos[1] * CELL_SIZE + 10))
        
        # Draw player
        player_rect = pygame.Rect(
            self.player_pos[0] * CELL_SIZE + 5, 
            self.player_pos[1] * CELL_SIZE + 5, 
            CELL_SIZE - 10, CELL_SIZE - 10
        )
        pygame.draw.ellipse(self.screen, YELLOW, player_rect)
        
        # Draw player direction indicator
        if len(self.player_trail) > 1:
            prev_x, prev_y = self.player_trail[-2]
            curr_x, curr_y = self.player_pos
            dx, dy = curr_x - prev_x, curr_y - prev_y
            if dx != 0 or dy != 0:
                center_x = self.player_pos[0] * CELL_SIZE + CELL_SIZE // 2
                center_y = self.player_pos[1] * CELL_SIZE + CELL_SIZE // 2
                end_x = center_x + dx * 15
                end_y = center_y + dy * 15
                pygame.draw.line(self.screen, ORANGE, (center_x, center_y), (end_x, end_y), 3)
        
        # Draw stats
        time_elapsed = (pygame.time.get_ticks() - self.start_time) // 1000
        stats_text = self.font.render(f'Moves: {self.moves} | Time: {time_elapsed}s', True, WHITE)
        self.screen.blit(stats_text, (10, 10))
        
        # Draw controls hint
        controls_text = self.font.render('Use hand gestures to move: Point fingers in direction', True, LIGHT_BLUE)
        self.screen.blit(controls_text, (WIDTH // 2 - controls_text.get_width() // 2, HEIGHT - 30))
        
        # Draw win message
        if self.game_won:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = self.big_font.render('MAZE SOLVED!', True, GREEN)
            stats_text = self.font.render(f'Completed in {self.moves} moves and {self.completion_time} seconds!', True, WHITE)
            restart_text = self.font.render('Press R to Play Again or Q to Quit', True, YELLOW)
            
            self.screen.blit(win_text, (WIDTH//2 - win_text.get_width()//2, HEIGHT//2 - 60))
            self.screen.blit(stats_text, (WIDTH//2 - stats_text.get_width()//2, HEIGHT//2))
            self.screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT//2 + 50))
        
        pygame.display.flip()

def detect_hand_direction(hand_landmarks, prev_landmarks=None):
    """
    Improved hand direction detection using multiple methods
    Returns direction (dx, dy) or None
    """
    landmarks = hand_landmarks.landmark
    
    # Method 1: Palm orientation (wrist to middle finger MCP)
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    dx_palm = middle_mcp.x - wrist.x
    dy_palm = middle_mcp.y - wrist.y
    
    # Method 2: Finger pointing direction
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    dx_finger = index_tip.x - index_mcp.x
    dy_finger = index_tip.y - index_mcp.y
    
    # Method 3: Hand center movement (if previous landmarks available)
    dx_movement, dy_movement = 0, 0
    if prev_landmarks:
        prev_wrist = prev_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        dx_movement = wrist.x - prev_wrist.x
        dy_movement = wrist.y - prev_wrist.y
    
    # Combine methods with weights
    dx = dx_palm * 0.5 + dx_finger * 0.3 + dx_movement * 0.2
    dy = dy_palm * 0.5 + dy_finger * 0.3 + dy_movement * 0.2
    
    # Dead zone to prevent jitter
    dead_zone = 0.08
    
    # Determine dominant direction
    if abs(dx) > abs(dy) and abs(dx) > dead_zone:
        if dx > 0:
            return (1, 0)  # Right
        else:
            return (-1, 0)  # Left
    elif abs(dy) > dead_zone:
        if dy > 0:
            return (0, 1)  # Down
        else:
            return (0, -1)  # Up
    
    return None

def detect_finger_gesture(hand_landmarks):
    """
    Alternative gesture detection using finger counting
    """
    landmarks = hand_landmarks.landmark
    
    # Check which fingers are extended
    fingers = []
    
    # Thumb (special case)
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_extended = thumb_tip.x < thumb_ip.x if thumb_tip.x < landmarks[mp_hands.HandLandmark.WRIST].x else thumb_tip.x > thumb_ip.x
    fingers.append(thumb_extended)
    
    # Other fingers
    for tip, pip in [(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                     (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                     (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                     (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)]:
        finger_tip = landmarks[tip]
        finger_pip = landmarks[pip]
        fingers.append(finger_tip.y < finger_pip.y)
    
    thumb, index, middle, ring, pinky = fingers
    
    # Gesture mapping
    if index and not middle and not ring and not pinky:  # Pointing index
        return RIGHT
    elif pinky and not index and not middle and not ring:  # Pointing pinky
        return LEFT
    elif all(fingers[1:]) and not thumb:  # All fingers except thumb (open hand)
        return DOWN
    elif not any(fingers):  # Fist
        return UP
    elif index and middle and not ring and not pinky:  # Victory sign
        return UP
    
    return None

def draw_gesture_info(frame, direction, gesture_type, landmarks=None):
    """Draw comprehensive gesture information on the camera feed"""
    h, w = frame.shape[:2]
    
    # Draw direction text
    direction_names = {
        (1, 0): "RIGHT",
        (-1, 0): "LEFT", 
        (0, 1): "DOWN",
        (0, -1): "UP",
        None: "CENTER"
    }
    direction_text = f"Direction: {direction_names[direction]}"
    cv2.putText(frame, direction_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw gesture type
    cv2.putText(frame, f"Method: {gesture_type}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw control instructions
    instructions = [
        "Maze Controls:",
        "Tilt hand OR use finger gestures:",
        "Index finger -> RIGHT",
        "Pinky finger -> LEFT",
        "Open hand -> DOWN", 
        "Fist/Victory -> UP",
        "Keep hand steady to stop"
    ]
    
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (w - 300, 40 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Draw direction indicator
    center_x, center_y = w // 2, h // 2
    if direction:
        dx, dy = direction
        end_x = center_x + int(dx * 80)
        end_y = center_y + int(dy * 80)
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 4, tipLength=0.3)
    
    # Draw center point
    cv2.circle(frame, (center_x, center_y), 8, (255, 0, 0), -1)
    cv2.circle(frame, (center_x, center_y), 12, (255, 255, 255), 2)
    
    # Draw hand bounding box if landmarks available
    if landmarks:
        x_coords = [landmark.x for landmark in landmarks.landmark]
        y_coords = [landmark.y for landmark in landmarks.landmark]
        
        min_x, max_x = int(min(x_coords) * w), int(max(x_coords) * w)
        min_y, max_y = int(min(y_coords) * h), int(max(y_coords) * h)
        
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

def main():
    game = MazeGame()
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting Improved Hand Gesture Controlled Maze Solver!")
    print("Use hand tilting OR finger gestures to control the ball")
    print("Press 'Q' in the camera window to quit")
    
    last_direction = None
    direction_stability = 0
    last_move_time = 0
    move_delay = 300  # milliseconds between moves
    prev_landmarks = None
    gesture_method = "tilt"  # Default method
    
    while True:
        current_time = pygame.time.get_ticks()
        
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
                    print("Game reset!")
                elif event.key == pygame.K_q:
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_t:
                    gesture_method = "tilt" if gesture_method == "fingers" else "fingers"
                    print(f"Switched to {gesture_method} gesture method")
        
        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue
            
        # Flip frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe Hands
        results = hands.process(rgb_frame)
        
        current_direction = None
        current_gesture_type = "None"
        
        # Detect hand and control game
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Detect gesture using selected method
                if gesture_method == "tilt":
                    detected_direction = detect_hand_direction(hand_landmarks, prev_landmarks)
                    current_gesture_type = "Tilt"
                else:
                    detected_direction = detect_finger_gesture(hand_landmarks)
                    current_gesture_type = "Fingers"
                
                # Add stability check to prevent flickering
                if detected_direction == last_direction:
                    direction_stability += 1
                else:
                    direction_stability = 0
                
                # Only move if direction is stable and enough time has passed
                if direction_stability >= 2 and current_time - last_move_time > move_delay:
                    current_direction = detected_direction
                    if current_direction:
                        game.move_player(current_direction)
                        last_move_time = current_time
                
                last_direction = detected_direction
                prev_landmarks = hand_landmarks
        
        # Draw gesture information
        draw_gesture_info(frame, current_direction, current_gesture_type, 
                         results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None)
        
        # Display method info
        method_text = f"Current Method: {gesture_method.upper()} (Press T to switch)"
        cv2.putText(frame, method_text, (20, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display camera feed
        cv2.imshow('Hand Gesture Control - Press Q to quit, T to switch method', frame)
        cv2.resizeWindow('Hand Gesture Control - Press Q to quit, T to switch method', 640, 480)
        
        # Draw game
        game.draw()
        
        # Control game speed
        game.clock.tick(FPS)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()